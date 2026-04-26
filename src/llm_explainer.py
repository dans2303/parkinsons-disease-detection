import os
from openai import OpenAI


FEATURE_DESCRIPTIONS = {
    "jitt": "jitter-related frequency instability",
    "jita": "absolute jitter, measuring cycle-to-cycle frequency variation",
    "RAP": "relative average perturbation, a jitter-based voice stability feature",
    "PPQ": "pitch perturbation quotient",
    "sPPQ": "smoothed pitch perturbation quotient",
    "vfo": "fundamental frequency variation",
    "shim": "shimmer-related amplitude instability",
    "shdb": "shimmer measured in decibels",
    "APQ": "amplitude perturbation quotient",
    "sAPQ": "smoothed amplitude perturbation quotient",
    "vAm": "amplitude variation",
    "ATRI": "amplitude tremor intensity index",
    "FTRI": "frequency tremor intensity index",
    "VTI": "voice turbulence index",
    "SPI": "soft phonation index",
    "NHR": "noise-to-harmonics ratio",
}


def build_prompt(prediction, probability, shap_df, explanation_style="Scientific + careful"):
    top_features = shap_df.head(5)

    feature_text = "\n".join(
        [
            f"- {row['feature']} ({FEATURE_DESCRIPTIONS.get(row['feature'], 'MDVP voice biomarker')}): "
            f"SHAP value={row['shap_value']:.3f}"
            for _, row in top_features.iterrows()
        ]
    )

    if explanation_style == "Simple + user-friendly":
        style_instruction = """
Explain the result in simple language for a general audience.
Avoid technical jargon where possible.
Use a calm and cautious tone.
"""
    else:
        style_instruction = """
Explain the result in a scientific but cautious way suitable for a research or academic audience.
Use biomedical machine learning language, but keep it understandable.
"""

    prompt = f"""
You are a biomedical AI assistant explaining a machine learning prediction based on MDVP voice biomarkers.

Model prediction: {prediction}
Predicted probability for Parkinson's Disease class: {probability:.3f}

Top contributing SHAP features:
{feature_text}

{style_instruction}

Important rules:
- Do NOT claim this is a medical diagnosis.
- Do NOT say the person has Parkinson's Disease.
- Say "the model predicts" or "the model output suggests".
- Explain that SHAP shows model behavior, not medical causality.
- Focus only on voice biomarker patterns.
- Do NOT mention unrelated symptoms such as movement, tremor, rigidity, or motor function unless directly tied to the listed voice features.
- Keep the explanation concise and professional.
"""

    return prompt.strip()


def generate_explanation(prompt):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "AI explanation is unavailable because OPENAI_API_KEY is not set. "
            "Please set your API key in the terminal before using this feature."
        )

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as error:
        return (
            "AI explanation could not be generated.\n\n"
            f"Error details: {error}"
        )