from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from rich import print
import joblib
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import RetrieverInput
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

# =====================================================
# APP
# =====================================================

app = FastAPI(title="AI Real Estate Recommendation API", version="1.0.0")

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODELS & FILES
# =====================================================

print("Loading ML Model...")

model = joblib.load("house_price_pipeline.pkl")

city_location_mapping = joblib.load("city_location_mapping.pkl")

location_lat_long = joblib.load("location_lat_long.pkl")

property_descriptions = joblib.load("property_descriptions.pkl")

property_metadata = joblib.load("property_metadata.pkl")

index = faiss.read_index("property_index.faiss")


print("All files loaded successfully.")

# =====================================================
# REQUEST MODEL
# =====================================================


class HouseRequest(BaseModel):
    area: float
    bedrooms: int

    city: str
    location: str

    gymnasium: int = 0
    swimming_pool: int = 0
    clubhouse: int = 0
    power_backup: int = 0
    car_parking: int = 0
    lift_available: int = 0
    ac: int = 0
    resale: int = 0


class ChatRequest(BaseModel):
    question: str


# =====================================================
# NLP PROPERTY RECOMMENDATION
# =====================================================


def get_recommendations(
    request_data: HouseRequest, predicted_price: float, top_k: int = 20
):

    query = f"""
    Property in {request_data.location}, {request_data.city}.
    Area {request_data.area} square feet.
    {request_data.bedrooms} bedrooms.
    Gymnasium {'yes' if request_data.gymnasium else 'no'}.
    Swimming Pool {'yes' if request_data.swimming_pool else 'no'}.
    Club House {'yes' if request_data.clubhouse else 'no'}.
    Power Backup {'yes' if request_data.power_backup else 'no'}.
    Car Parking {'yes' if request_data.car_parking else 'no'}.
    Lift {'yes' if request_data.lift_available else 'no'}.
    AC {'yes' if request_data.ac else 'no'}.
    """

    query_embedding = embeddings_rag.encode(query, convert_to_numpy=True).astype(
        np.float32
    )

    faiss.normalize_L2(query_embedding.reshape(1, -1))

    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)

    recommendations = []

    min_price = predicted_price * 0.80
    max_price = predicted_price * 1.20

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        property_info = property_metadata[idx]

        property_price = property_info.get("price", property_info.get("Price", 0))

        # Budget filtering
        if not (min_price <= property_price <= max_price):
            continue

        recommendations.append(
            {
                "city": property_info.get("city"),
                "location": property_info.get("location"),
                "price": round(float(property_price), 2),
                "area": property_info.get("area"),
                "bedrooms": property_info.get("bedrooms"),
                "gymnasium": property_info.get("gymnasium", 0),
                "swimming_pool": property_info.get("swimming_pool", 0),
                "clubhouse": property_info.get("clubhouse", 0),
                "power_backup": property_info.get("power_backup", 0),
                "car_parking": property_info.get("car_parking", 0),
                "lift_available": property_info.get("lift_available", 0),
                "ac": property_info.get("ac", 0),
                "similarity_score": round(float(score) * 100, 2),
            }
        )

        if len(recommendations) >= 5:
            break

    return recommendations


# =====================================================
# HOME
# =====================================================


@app.get("/")
def home():
    return {"message": "AI Real Estate Recommendation API Running"}


# =====================================================
# GET CITIES
# =====================================================


@app.get("/cities")
def get_cities():

    return {"cities": sorted(city_location_mapping.keys())}


# =====================================================
# GET LOCATIONS
# =====================================================


@app.get("/locations/{city}")
def get_locations(city: str):

    return {"locations": sorted(city_location_mapping.get(city, []))}


# =====================================================
# PRICE PREDICTION + RECOMMENDATION
# =====================================================


@app.post("/predict")
def predict(data: HouseRequest):

    if data.location not in location_lat_long:

        raise HTTPException(
            status_code=400, detail=f"Location '{data.location}' not found"
        )

    latitude = location_lat_long[data.location]["Latitude"]
    longitude = location_lat_long[data.location]["Longitude"]

    input_df = pd.DataFrame(
        [
            {
                "Area": data.area,
                "No. of Bedrooms": data.bedrooms,
                "Latitude": latitude,
                "Longitude": longitude,
                "Resale": data.resale,
                "Gymnasium": data.gymnasium,
                "SwimmingPool": data.swimming_pool,
                "ClubHouse": data.clubhouse,
                "PowerBackup": data.power_backup,
                "CarParking": data.car_parking,
                "LiftAvailable": data.lift_available,
                "AC": data.ac,
                "City": data.city,
                "Location": data.location,
            }
        ]
    )

    # Feature Engineering

    input_df["Area_per_Bedroom"] = input_df["Area"] / input_df["No. of Bedrooms"]

    input_df["Lat_Long"] = input_df["Latitude"] * input_df["Longitude"]

    input_df["Luxury_Score"] = (
        input_df["Gymnasium"]
        + input_df["SwimmingPool"]
        + input_df["ClubHouse"]
        + input_df["PowerBackup"]
        + input_df["LiftAvailable"]
    )

    # Missing columns used during training

    for col in ["JoggingTrack", "Wardrobe", "Wifi"]:
        input_df[col] = 0

    # Price Prediction

    prediction_log = model.predict(input_df)[0]

    predicted_price = float(np.expm1(prediction_log))

    # Property Recommendation

    recommendations = get_recommendations(
        request_data=data, predicted_price=predicted_price
    )

    return {
        "predicted_price": round(predicted_price, 2),
        "recommended_properties": recommendations,
    }


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are HomeAI, a professional and trusted real estate consultant for the Indian property market.

Your goal is to help users discover, compare, evaluate, and understand properties in a natural, human-like manner.

PERSONALITY:
- Polite, professional, knowledgeable, and helpful.
- Speak like an experienced property advisor.
- Be conversational and friendly.
- Build trust through accurate information.
- Never sound robotic.

STRICT DATA RULES:
1. Use ONLY the property information available in CONTEXT.
2. Never invent:
   - Property details
   - Amenities
   - Prices
   - Sizes
   - Locations
   - Builder information
3. If information is unavailable, say:
   "I don't have that information available at the moment."
4. Never mention:
   - Context
   - Database
   - Retrieval
   - Documents
   - Embeddings
   - Vectors
   - Internal systems
   - Source data

CONVERSATION RULES:
1. If the user greets you:
   - Respond naturally and politely.
   - Offer assistance with property search.

Example:
User: Hi
Assistant:
Hello! 👋
I'd be happy to help you find the right property. Please share your preferred location, budget, property type, or any specific requirements.

2. If the user's requirements are incomplete:
   Politely ask follow-up questions.

Example:
User: I want a flat.
Assistant:
I'd be glad to help. Could you share:
• Preferred city or locality
• Budget range
• Number of bedrooms
• Any amenities you would like

3. If matching properties are available:
   - Show the most relevant options first.
   - Clearly highlight:
     • Property Name
     • Location
     • Property Type
     • Bedrooms
     • Area
     • Price
     • Key Amenities
   - Use clean formatting.
   - Explain why the property may suit the user's needs.

PROPERTY PRESENTATION FORMAT:

🏠 Property Name

📍 Location: [Location]
🏡 Type: [Property Type]
🛏 Bedrooms: [Bedrooms]
📐 Area: [Area]
💰 Price: [Price]

✨ Key Amenities:
• Amenity 1
• Amenity 2
• Amenity 3

💡 Why it may suit you:
[Short recommendation]

4. If multiple relevant properties exist:
   - Compare them briefly.
   - Mention strengths and differences.
   - Help the user make a decision.

Example:
For larger families, Property A offers more space.
If budget is your priority, Property B provides better value.

5. If no exact match exists:
   - Recommend the closest available properties.
   - Explain why they are similar.
   - Never claim they exactly match.

Example:
I couldn't find an exact match for your requirements. However, these options are quite similar and may be worth considering.

6. If no suitable properties are available:
   Say:
   "I couldn't find a suitable property matching your requirements at the moment. If you'd like, I can help you explore nearby locations or adjust the search criteria."

REAL ESTATE GUIDELINES:
- Use Indian real estate terminology.
- Use ₹ for prices.
- Mention area in sq ft whenever available.
- Prioritize relevance over quantity.
- Focus on helping users make informed decisions.

COMMUNICATION STYLE:
- Keep responses concise but informative.
- Use bullet points where appropriate.
- Avoid long paragraphs.
- Be respectful and professional.
- Always aim to guide the user toward the best available option.

=================================================
SUPPORTED CITIES
=================================================

The platform currently supports ONLY:

• Bangalore
• Chennai
• Delhi
• Hyderabad
• Kolkata
• Mumbai

Never mention any city outside these six cities.

=================================================
GREETING RULES
=================================================

When the user greets you and has NOT specified a preferred city:

Respond:

Hello! 👋

I'd be happy to help you find the right property.

Which city are you interested in?

• Bangalore
• Chennai
• Delhi
• Hyderabad
• Kolkata
• Mumbai

You may also share:
• Budget
• Property Type
• Number of Bedrooms
• Preferred Amenities

IMPORTANT:
Do NOT recommend any property during a greeting.
Do NOT assume a city.
Do NOT show property listings until the user selects a city.

=================================================
CITY PREFERENCE MEMORY
=================================================

If the user specifies a city, treat it as the preferred city for the remainder of the conversation unless the user changes it.

Examples:

User: Show me properties in Chennai.
Preferred City = Chennai

User: Show me 3 BHK apartments.
Use Chennai properties only.

User: Show me luxury properties.
Use Chennai properties only.

User: Compare two options.
Use Chennai properties only.

Do not switch cities unless the user explicitly requests another city.

=================================================
STRICT CITY FILTERING
=================================================

If a preferred city exists:

ONLY recommend properties from that city.

NEVER recommend properties from other cities.

Example:

Preferred City = Chennai

Allowed:
✓ Chennai properties

Forbidden:
✗ Bangalore properties
✗ Mumbai properties
✗ Delhi properties
✗ Hyderabad properties
✗ Kolkata properties

If no matching properties exist in the preferred city:

Respond:

"I couldn't find a suitable property in Chennai matching your requirements."

Do NOT suggest properties from other cities unless the user explicitly asks for alternatives.

=================================================
ZERO HALLUCINATION POLICY
=================================================

Every property recommendation must exist in CONTEXT.

Never create:

- Property names
- Cities
- Locations
- Amenities
- Prices
- Areas
- BHK counts

If a property field is missing:

"I don't have that information available at the moment."

Never estimate.

Never assume.

Never fabricate.

Never combine information from multiple properties.

Only display facts that appear exactly in CONTEXT.

=================================================
NO CROSS-CITY RECOMMENDATIONS
=================================================

If the user asks:

"Show properties in Chennai"

Only show Chennai properties.

If the user later asks:

"Show luxury properties"

Still show only Chennai luxury properties.

Do not display:
- Mumbai
- Bangalore
- Delhi
- Hyderabad
- Kolkata

unless the user explicitly changes the city.

Your responsibility is to act like a trusted real estate consultant and provide accurate recommendations using only available property information.
""",
        ),
        (
            "user",
            """
CONTEXT:
{context}

QUESTION:
{question}
""",
        ),
    ]
)

embeddings_rag = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "property_index", embeddings_rag, allow_dangerous_deserialization=True
)

retriever_rag = db.as_retriever(
    search_type="similarity", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

llm = ChatMistralAI(model="mistral-medium-latest")

message = []

from langchain_core.messages import HumanMessage, AIMessage


@app.post("/chat")
def chat(data: ChatRequest):

    query = data.question

    docs = retriever_rag.invoke(query)

    message.append(HumanMessage(query))

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": message})

    response = llm.invoke(final_prompt)

    message.append(response)

    print("\n===== CHAT HISTORY =====")

    print(message)
    print("========================\n")

    return {"answer": response.content, "sources": [doc.metadata for doc in docs]}


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
