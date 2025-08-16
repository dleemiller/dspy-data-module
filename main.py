import dspy
from data_builder import DataBuilder

lm_studio_model_id = "huggingfacetb_smollm3-3b"

try:
    lm_studio_client = dspy.LM(
        f"openai/{lm_studio_model_id}",
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
    )
    dspy.settings.configure(lm=lm_studio_client)
    print(f"✅ Configured to use LM Studio with model: {lm_studio_model_id}")

except Exception as e:
    print(f"⚠️ Could not configure LM Studio: {e}")
    exit()


# --- 2. Define a Signature and Reward Function ---
class ProductReview(dspy.Signature):
    """Assess a product review."""

    review_text: str = dspy.InputField()
    sentiment_score: int = dspy.OutputField(desc="An integer score from -1 to 1")
    is_legitimate: bool = dspy.OutputField(desc="True if the review is legitimate")


def sentiment_reward_function(args, pred: dspy.Prediction) -> float:
    """Rewards predictions that have a sentiment score."""
    try:
        score = int(pred.sentiment_score)
        return 1.0 if score in [-1, 0, 1] else 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


predictor = dspy.ChainOfThought(ProductReview)
builder = DataBuilder(
    predictor=predictor, output_dir="review_dataset/", reward_fn=sentiment_reward_function
)

reviews_to_process = [
    {"review_text": "This product is a total scam. It broke in 5 minutes."},
    {"review_text": "I absolutely love this thing! It's the best purchase I've made all year."},
    {"review_text": "It's okay. Not great, not terrible. Does the job."},
    {"review_text": "I'm not sure if this is a real product or a prank."},
]

print("\n--- Generating 3 responses for each of the 4 reviews (12 total calls) ---")
predictions = builder(examples=reviews_to_process, n=3, num_threads=4)

print(f"\n✅ Successfully generated {len(predictions)} predictions.")

print("\n--- Generating 5 responses for a single review ---")
single_review = {"review_text": "This is the pinnacle of human engineering."}
predictions_single = builder(examples=single_review, n=5, num_threads=5)

print(f"\n✅ Successfully generated {len(predictions_single)} predictions for the single review.")
