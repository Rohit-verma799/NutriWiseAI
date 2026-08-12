# =============================================================================
# NutriWise AI – Personalized Nutrition Coach
# Powered by IBM watsonx.ai Granite Models
# Multi-Agent Architecture: Nutrition Knowledge | Diet Planner | Health Advisory | Meal Analysis
# =============================================================================

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# =============================================================================
# IBM watsonx.ai Configuration
# Credentials are read from environment variables for security
# =============================================================================
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_MODEL_ID   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

# IBM IAM token endpoint
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# IBM watsonx.ai inference endpoint
WATSONX_INFERENCE_URL = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"

# Model candidate list with fallbacks in case a specific model is retired or unavailable in the region
MODEL_CANDIDATES = list(dict.fromkeys([
    WATSONX_MODEL_ID,
    "ibm/granite-4-h-small",
    "meta-llama/llama-3-3-70b-instruct",
    "mistralai/mistral-small-3-1-24b-instruct-2503"
]))


# =============================================================================
# IBM watsonx.ai Core Function
# This function is the single gateway to IBM watsonx.ai models.
# All four agents call this function to generate AI responses.
# =============================================================================
def get_iam_token():
    """Retrieve an IAM access token using the IBM Cloud API key."""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY
    }
    response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def generate_response(prompt: str, max_tokens: int = 800) -> str:
    """
    Core function that calls IBM watsonx.ai models to generate a response.
    Supports fallback models if the primary model is retired or unsupported in the current region.

    Args:
        prompt   : The instruction/question string sent to the model.
        max_tokens: Maximum number of tokens in the generated response.

    Returns:
        Generated text string from the foundation model.
    """
    try:
        # Step 1: Obtain IBM IAM access token
        token = get_iam_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        last_error = ""

        # Step 2: Try primary model and fallbacks if necessary
        for model_id in MODEL_CANDIDATES:
            payload = {
                "model_id": model_id,
                "project_id": WATSONX_PROJECT_ID,
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": 10,
                    "repetition_penalty": 1.1
                }
            }

            # Step 3: Call IBM watsonx.ai inference endpoint
            response = requests.post(
                WATSONX_INFERENCE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result["results"][0]["generated_text"].strip()
                return generated_text

            # Parse exact JSON error message from Watsonx if available
            try:
                err_data = response.json()
                err_msg = err_data.get("errors", [{}])[0].get("message", response.text)
            except Exception:
                err_msg = response.text

            last_error = f"{response.status_code} Error: {err_msg}"

            # If 404 (model not found/deprecated) or 400 (model function unsupported), try next fallback
            if response.status_code in (404, 400):
                continue
            else:
                break

        return f"❌ IBM watsonx.ai API error: {last_error}"

    except requests.exceptions.HTTPError as e:
        return f"❌ IBM watsonx.ai API error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error calling IBM watsonx.ai: {str(e)}"


# =============================================================================
# AGENT 1: Nutrition Knowledge Agent
# Answers general nutrition questions using IBM Granite models.
# =============================================================================
def nutrition_knowledge_agent(question: str) -> str:
    """
    Agent 1 – Nutrition Knowledge Agent
    Routes user nutrition questions to IBM watsonx.ai Granite for educational responses.
    """
    prompt = f"""You are NutriWise, an expert nutritionist and dietitian AI assistant powered by IBM watsonx.ai.
Answer the following nutrition question clearly, accurately, and helpfully.
Provide structured information including benefits, key nutrients, and practical tips.
Keep the response informative yet concise (under 300 words).

Question: {question}

Answer:"""

    # IBM watsonx.ai Granite model generates the nutrition response
    return generate_response(prompt, max_tokens=600)


# =============================================================================
# AGENT 2: Diet Planner Agent
# Creates personalized meal plans based on user profile using IBM Granite models.
# =============================================================================
def diet_planner_agent(age, gender, height, weight, dietary_pref, activity_level, fitness_goal) -> str:
    """
    Agent 2 – Diet Planner Agent
    Generates a personalized daily meal plan based on user demographics and fitness goals.
    Uses IBM watsonx.ai Granite to produce customized meal recommendations.
    """
    prompt = f"""You are NutriWise Diet Planner, an expert AI nutritionist powered by IBM watsonx.ai.
Create a detailed, personalized daily meal plan for the following individual:

User Profile:
- Age: {age} years
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Dietary Preference: {dietary_pref}
- Activity Level: {activity_level}
- Fitness Goal: {fitness_goal}

Generate a structured meal plan in this exact format:

## Daily Nutritional Targets
- **Estimated Calorie Target:** [calories] kcal/day
- **Protein:** [grams]g/day
- **Carbohydrates:** [grams]g/day
- **Fats:** [grams]g/day

## 🌅 Breakfast
[List 3-4 food items with approximate portions]

## ☀️ Lunch
[List 3-4 food items with approximate portions]

## 🍎 Evening Snack
[List 2-3 healthy snack options]

## 🌙 Dinner
[List 3-4 food items with approximate portions]

## 💡 Key Nutrition Tips
[3 personalized tips based on the fitness goal]

Meal Plan:"""

    # IBM watsonx.ai Granite model generates the personalized meal plan
    return generate_response(prompt, max_tokens=900)


# =============================================================================
# AGENT 3: Health Advisory Agent
# Provides disease-specific dietary recommendations using IBM Granite models.
# =============================================================================
def health_advisory_agent(conditions: list) -> str:
    """
    Agent 3 – Health Advisory Agent
    Generates dietary and lifestyle recommendations for selected health conditions.
    Uses IBM watsonx.ai Granite to provide evidence-based advisory content.
    Always includes a medical disclaimer.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Health"

    prompt = f"""You are NutriWise Health Advisor, a specialized AI health assistant powered by IBM watsonx.ai.
Provide comprehensive dietary and lifestyle recommendations for the following health condition(s): {conditions_str}

Generate a structured health advisory in this exact format:

## 🥗 Foods to Include
[List 6-8 beneficial foods with brief reasons]

## 🚫 Foods to Avoid
[List 6-8 foods/ingredients to avoid with reasons]

## 🏃 Healthy Habits to Follow
[List 5-6 lifestyle habits and routines]

## 💊 Lifestyle Recommendations
[List 4-5 broader lifestyle changes and recommendations]

## ⚠️ Important Notes
[Condition-specific important nutritional notes]

Health Advisory:"""

    # IBM watsonx.ai Granite model generates condition-specific health recommendations
    response = generate_response(prompt, max_tokens=900)

    # Always append the medical disclaimer (Agent 3 requirement)
    disclaimer = (
        "\n\n---\n"
        "⚕️ **Disclaimer:** This information is for educational purposes only. "
        "Always consult a qualified healthcare professional or registered dietitian "
        "before making significant changes to your diet, especially if you have a medical condition."
    )
    return response + disclaimer


# =============================================================================
# AGENT 4: Meal Analysis Agent
# Analyzes user-entered meals and provides AI-powered nutritional feedback.
# =============================================================================
def meal_analysis_agent(meal_text: str) -> str:
    """
    Agent 4 – Meal Analysis Agent
    Analyzes free-text meal descriptions using IBM watsonx.ai Granite.
    Estimates nutritional quality, identifies strengths/deficiencies, and suggests improvements.
    """
    prompt = f"""You are NutriWise Meal Analyzer, an expert AI dietitian powered by IBM watsonx.ai.
Analyze the following meal description and provide a comprehensive nutritional assessment.

Meal Description:
{meal_text}

Provide a structured analysis in this exact format:

## 📊 Nutritional Quality Score
Rate overall nutritional quality: [X/10] — [brief justification]

## ✅ Nutritional Strengths
[List 4-5 positive aspects of this meal]

## ⚠️ Nutritional Deficiencies
[List 4-5 nutritional gaps or concerns]

## 🔄 Healthier Alternatives
[Suggest specific ingredient substitutions or additions]

## 💡 Improvement Recommendations
[List 4-5 actionable suggestions to make the meals more balanced]

## 📈 Estimated Macronutrient Balance
- Proteins: [Low/Moderate/High]
- Carbohydrates: [Low/Moderate/High]
- Healthy Fats: [Low/Moderate/High]
- Fiber: [Low/Moderate/High]
- Micronutrients: [key vitamins/minerals present or missing]

Meal Analysis:"""

    # IBM watsonx.ai Granite model analyzes the meal and generates personalized feedback
    return generate_response(prompt, max_tokens=900)


# =============================================================================
# AGENT ORCHESTRATOR
# Routes incoming requests to the appropriate specialized agent.
# =============================================================================
def orchestrate(agent_name: str, data: dict) -> str:
    """
    Orchestrator function that routes requests to the correct agent.

    Agents:
      - nutrition_knowledge : Agent 1 – answers nutrition questions
      - diet_planner        : Agent 2 – creates personalized meal plans
      - health_advisory     : Agent 3 – disease-specific dietary guidance
      - meal_analysis       : Agent 4 – analyzes meals and provides feedback
    """
    if agent_name == "nutrition_knowledge":
        return nutrition_knowledge_agent(data.get("question", ""))

    elif agent_name == "diet_planner":
        return diet_planner_agent(
            age=data.get("age", 25),
            gender=data.get("gender", "Male"),
            height=data.get("height", 170),
            weight=data.get("weight", 70),
            dietary_pref=data.get("dietary_pref", "Vegetarian"),
            activity_level=data.get("activity_level", "Moderately Active"),
            fitness_goal=data.get("fitness_goal", "General Wellness")
        )

    elif agent_name == "health_advisory":
        return health_advisory_agent(data.get("conditions", []))

    elif agent_name == "meal_analysis":
        return meal_analysis_agent(data.get("meal_text", ""))

    else:
        return "❌ Unknown agent. Please select a valid feature."


# =============================================================================
# HTML TEMPLATE — Full single-page application with Bootstrap 5
# All HTML is embedded inside app.py using render_template_string()
# =============================================================================
BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NutriWise AI – Personalized Nutrition Coach</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --primary: #1a7f5a;
      --primary-dark: #145f44;
      --primary-light: #e8f5f0;
      --accent: #2ecc8f;
      --sidebar-width: 260px;
      --text-dark: #1e2d24;
      --text-muted: #6b7c73;
      --card-radius: 14px;
      --shadow: 0 4px 24px rgba(26,127,90,0.09);
    }
    * { font-family: 'Inter', sans-serif; }
    body { background: #f4f8f6; min-height: 100vh; color: var(--text-dark); }

    /* ---- Sidebar ---- */
    #sidebar {
      width: var(--sidebar-width);
      min-height: 100vh;
      background: linear-gradient(160deg, #0f4c33 0%, #1a7f5a 60%, #25a876 100%);
      position: fixed; top: 0; left: 0; z-index: 100;
      display: flex; flex-direction: column;
      box-shadow: 4px 0 20px rgba(0,0,0,0.15);
      transition: transform 0.3s ease;
    }
    .sidebar-brand {
      padding: 28px 24px 18px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
    }
    .sidebar-brand h4 { color: #fff; font-weight: 700; font-size: 1.15rem; margin: 0; letter-spacing: -0.3px; }
    .sidebar-brand p  { color: rgba(255,255,255,0.65); font-size: 0.72rem; margin: 0; }
    .sidebar-logo { width: 42px; height: 42px; background: rgba(255,255,255,0.15);
      border-radius: 12px; display: flex; align-items: center; justify-content: center;
      margin-bottom: 10px; font-size: 1.4rem; }
    .nav-section-label {
      font-size: 0.65rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
      color: rgba(255,255,255,0.45); padding: 18px 24px 6px;
    }
    .nav-link-custom {
      display: flex; align-items: center; gap: 11px;
      padding: 11px 24px; color: rgba(255,255,255,0.82);
      text-decoration: none; font-size: 0.875rem; font-weight: 500;
      border-radius: 0; transition: all 0.2s; position: relative;
    }
    .nav-link-custom:hover, .nav-link-custom.active {
      background: rgba(255,255,255,0.12); color: #fff;
    }
    .nav-link-custom.active::before {
      content: ''; position: absolute; left: 0; top: 0; bottom: 0;
      width: 3px; background: #2ecc8f; border-radius: 0 3px 3px 0;
    }
    .nav-link-custom i { font-size: 1rem; width: 20px; text-align: center; }
    .sidebar-footer {
      margin-top: auto; padding: 16px 20px;
      border-top: 1px solid rgba(255,255,255,0.1);
    }
    .ibm-badge {
      background: rgba(255,255,255,0.1); border-radius: 8px;
      padding: 8px 12px; display: flex; align-items: center; gap: 8px;
    }
    .ibm-badge span { color: rgba(255,255,255,0.8); font-size: 0.72rem; }
    .ibm-dot { width: 8px; height: 8px; background: #2ecc8f; border-radius: 50%;
      animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

    /* ---- Main Content ---- */
    #main-content {
      margin-left: var(--sidebar-width);
      min-height: 100vh;
      padding: 0;
    }
    .top-bar {
      background: #fff; border-bottom: 1px solid #e5ede9;
      padding: 14px 32px; display: flex; align-items: center;
      justify-content: space-between; position: sticky; top: 0; z-index: 50;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .top-bar h5 { font-weight: 600; color: var(--text-dark); margin: 0; font-size: 1rem; }
    .page-content { padding: 32px; }

    /* ---- Cards ---- */
    .card-custom {
      background: #fff; border-radius: var(--card-radius);
      box-shadow: var(--shadow); border: 1px solid #e5ede9;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .card-custom:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(26,127,90,0.13); }
    .agent-icon-box {
      width: 52px; height: 52px; border-radius: 14px;
      display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
    }
    .icon-green  { background: #e8f5f0; color: #1a7f5a; }
    .icon-blue   { background: #e8f0fa; color: #2563eb; }
    .icon-orange { background: #fff4e8; color: #d97706; }
    .icon-purple { background: #f3e8ff; color: #7c3aed; }

    /* ---- Forms & Inputs ---- */
    .form-control, .form-select {
      border: 1.5px solid #d4e4dc; border-radius: 10px;
      font-size: 0.9rem; padding: 10px 14px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .form-control:focus, .form-select:focus {
      border-color: var(--primary); box-shadow: 0 0 0 3px rgba(26,127,90,0.12);
    }
    textarea.form-control { resize: vertical; min-height: 120px; }
    .btn-primary-custom {
      background: linear-gradient(135deg, #1a7f5a, #25a876);
      color: #fff; border: none; border-radius: 10px;
      padding: 11px 28px; font-weight: 600; font-size: 0.9rem;
      transition: all 0.2s; cursor: pointer;
    }
    .btn-primary-custom:hover {
      background: linear-gradient(135deg, #145f44, #1a7f5a);
      transform: translateY(-1px); box-shadow: 0 4px 14px rgba(26,127,90,0.3);
    }
    .btn-primary-custom:disabled { opacity: 0.65; cursor: not-allowed; transform: none; }

    /* ---- Chat / Response Area ---- */
    .response-box {
      background: var(--primary-light); border: 1.5px solid #b8dece;
      border-radius: var(--card-radius); padding: 22px 24px;
      min-height: 60px; display: none;
    }
    .response-box.visible { display: block; }
    .response-content { white-space: pre-wrap; line-height: 1.75; font-size: 0.92rem; }
    .response-content h2 { font-size: 1rem; font-weight: 700; color: var(--primary-dark); margin-top: 16px; }
    .response-content strong { color: var(--primary-dark); }
    .loading-spinner {
      display: flex; align-items: center; gap: 12px;
      color: var(--primary); font-weight: 500; font-size: 0.9rem;
    }
    .spinner-dot {
      width: 10px; height: 10px; border-radius: 50%; background: var(--primary);
      animation: bounce 1.4s infinite ease-in-out both;
    }
    .spinner-dot:nth-child(1) { animation-delay: -0.32s; }
    .spinner-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
      0%,80%,100%{transform:scale(0);} 40%{transform:scale(1);}
    }

    /* ---- Chat Messages ---- */
    .chat-messages { max-height: 420px; overflow-y: auto; padding: 6px 0; }
    .chat-msg {
      margin-bottom: 14px; display: flex; gap: 10px; align-items: flex-start;
    }
    .chat-msg.user { flex-direction: row-reverse; }
    .chat-bubble {
      max-width: 80%; padding: 12px 16px; border-radius: 14px;
      font-size: 0.875rem; line-height: 1.65; white-space: pre-wrap;
    }
    .chat-bubble.user-bubble {
      background: linear-gradient(135deg, #1a7f5a, #25a876);
      color: #fff; border-bottom-right-radius: 4px;
    }
    .chat-bubble.ai-bubble {
      background: #fff; border: 1.5px solid #d4e4dc;
      color: var(--text-dark); border-bottom-left-radius: 4px;
    }
    .chat-avatar {
      width: 34px; height: 34px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.9rem; flex-shrink: 0;
    }
    .avatar-ai { background: var(--primary-light); color: var(--primary); }
    .avatar-user { background: linear-gradient(135deg, #1a7f5a, #25a876); color: #fff; }

    /* ---- Hero Section ---- */
    .hero-gradient {
      background: linear-gradient(135deg, #0f4c33 0%, #1a7f5a 50%, #25a876 100%);
      border-radius: 18px; color: #fff; padding: 40px;
      position: relative; overflow: hidden;
    }
    .hero-gradient::after {
      content: ''; position: absolute; right: -20px; top: -20px;
      width: 180px; height: 180px; border-radius: 50%;
      background: rgba(255,255,255,0.05);
    }
    .stat-card {
      background: #fff; border-radius: 12px; padding: 18px 20px;
      border: 1.5px solid #e5ede9; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
    .stat-label  { font-size: 0.75rem; color: var(--text-muted); font-weight: 500; }

    /* ---- Condition Badges ---- */
    .condition-check {
      background: #f4f8f6; border: 1.5px solid #d4e4dc;
      border-radius: 10px; padding: 10px 16px; cursor: pointer;
      transition: all 0.2s; display: flex; align-items: center; gap: 9px;
      font-size: 0.875rem; font-weight: 500;
    }
    .condition-check:hover { border-color: var(--primary); background: var(--primary-light); }
    .condition-check input:checked + label, .condition-check.selected {
      border-color: var(--primary); background: var(--primary-light); color: var(--primary);
    }

    /* ---- About Page ---- */
    .arch-step {
      background: #fff; border-radius: 12px; padding: 20px;
      border-left: 4px solid var(--primary); margin-bottom: 14px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .tech-pill {
      display: inline-block; background: var(--primary-light);
      color: var(--primary); border-radius: 20px;
      padding: 4px 12px; font-size: 0.78rem; font-weight: 600; margin: 3px;
    }

    /* ---- Page visibility ---- */
    .page { display: none; }
    .page.active { display: block; }

    /* ---- Responsive ---- */
    @media (max-width: 768px) {
      #sidebar { transform: translateX(-100%); }
      #sidebar.open { transform: translateX(0); }
      #main-content { margin-left: 0; }
      .hero-gradient { padding: 24px; }
      .page-content { padding: 16px; }
    }
  </style>
</head>
<body>

<!-- ========== SIDEBAR ========== -->
<nav id="sidebar">
  <div class="sidebar-brand">
    <div class="sidebar-logo">🥗</div>
    <h4>NutriWise AI</h4>
    <p>Personalized Nutrition Coach</p>
  </div>

  <div class="nav-section-label">Navigation</div>
  <a href="#" class="nav-link-custom active" onclick="showPage('home',this); return false;">
    <i class="bi bi-house-door"></i> Home
  </a>
  <a href="#" class="nav-link-custom" onclick="showPage('chat',this); return false;">
    <i class="bi bi-chat-dots"></i> Nutrition Chat
  </a>
  <a href="#" class="nav-link-custom" onclick="showPage('planner',this); return false;">
    <i class="bi bi-calendar3"></i> Diet Planner
  </a>
  <a href="#" class="nav-link-custom" onclick="showPage('advisor',this); return false;">
    <i class="bi bi-heart-pulse"></i> Health Advisor
  </a>
  <a href="#" class="nav-link-custom" onclick="showPage('analyzer',this); return false;">
    <i class="bi bi-search"></i> Meal Analyzer
  </a>

  <div class="nav-section-label">Info</div>
  <a href="#" class="nav-link-custom" onclick="showPage('about',this); return false;">
    <i class="bi bi-info-circle"></i> About
  </a>

  <div class="sidebar-footer">
    <div class="ibm-badge">
      <div class="ibm-dot"></div>
      <span>Powered by IBM watsonx.ai</span>
    </div>
  </div>
</nav>

<!-- ========== MAIN CONTENT ========== -->
<div id="main-content">
  <div class="top-bar">
    <div class="d-flex align-items-center gap-3">
      <button class="btn btn-sm d-md-none" onclick="toggleSidebar()">
        <i class="bi bi-list fs-5"></i>
      </button>
      <h5 id="page-title">🏠 Welcome to NutriWise AI</h5>
    </div>
    <span class="badge rounded-pill text-bg-success px-3 py-2" style="font-size:0.72rem;">
      <i class="bi bi-circle-fill me-1" style="font-size:0.5rem;"></i>
      IBM Granite Model Active
    </span>
  </div>

  <div class="page-content">

    <!-- ===== PAGE: HOME ===== -->
    <div id="page-home" class="page active">
      <div class="hero-gradient mb-4">
        <div class="row align-items-center">
          <div class="col-md-8">
            <h1 class="fw-700 mb-2" style="font-weight:700;font-size:2rem;">NutriWise AI 🥗</h1>
            <p class="mb-3" style="opacity:0.9;font-size:1.05rem;">
              Your AI-powered personalized nutrition coach, built on IBM watsonx.ai Granite Models.
              Get expert dietary guidance, personalized meal plans, and health-specific advice — all from intelligent AI agents.
            </p>
            <button class="btn btn-light fw-600 px-4" onclick="showPage('chat', document.querySelectorAll('.nav-link-custom')[1])">
              Get Started <i class="bi bi-arrow-right ms-1"></i>
            </button>
          </div>
          <div class="col-md-4 text-center d-none d-md-block">
            <div style="font-size:5rem;">🧠🥦</div>
          </div>
        </div>
      </div>

      <!-- Stats -->
      <div class="row g-3 mb-4">
        <div class="col-6 col-md-3">
          <div class="stat-card">
            <div class="stat-number">4</div>
            <div class="stat-label">AI Agents</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="stat-card">
            <div class="stat-number">IBM</div>
            <div class="stat-label">Granite Powered</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="stat-card">
            <div class="stat-number">∞</div>
            <div class="stat-label">Personalized Plans</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="stat-card">
            <div class="stat-number">6</div>
            <div class="stat-label">Health Conditions</div>
          </div>
        </div>
      </div>

      <!-- Agent Cards -->
      <h5 class="fw-600 mb-3">🤖 Meet Your AI Agents</h5>
      <div class="row g-3">
        <div class="col-md-6">
          <div class="card-custom p-4" onclick="showPage('chat', document.querySelectorAll('.nav-link-custom')[1])" style="cursor:pointer;">
            <div class="d-flex align-items-center gap-3 mb-3">
              <div class="agent-icon-box icon-green"><i class="bi bi-chat-quote"></i></div>
              <div>
                <h6 class="fw-600 mb-0">Nutrition Knowledge Agent</h6>
                <small class="text-muted">Agent 1</small>
              </div>
            </div>
            <p class="text-muted mb-0" style="font-size:0.875rem;">
              Ask any nutrition question — from food benefits to vitamins. Get instant AI-powered answers from IBM Granite.
            </p>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card-custom p-4" onclick="showPage('planner', document.querySelectorAll('.nav-link-custom')[2])" style="cursor:pointer;">
            <div class="d-flex align-items-center gap-3 mb-3">
              <div class="agent-icon-box icon-blue"><i class="bi bi-calendar-check"></i></div>
              <div>
                <h6 class="fw-600 mb-0">Diet Planner Agent</h6>
                <small class="text-muted">Agent 2</small>
              </div>
            </div>
            <p class="text-muted mb-0" style="font-size:0.875rem;">
              Get a fully personalized daily meal plan tailored to your body, goals, and dietary preferences.
            </p>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card-custom p-4" onclick="showPage('advisor', document.querySelectorAll('.nav-link-custom')[3])" style="cursor:pointer;">
            <div class="d-flex align-items-center gap-3 mb-3">
              <div class="agent-icon-box icon-orange"><i class="bi bi-heart-pulse"></i></div>
              <div>
                <h6 class="fw-600 mb-0">Health Advisory Agent</h6>
                <small class="text-muted">Agent 3</small>
              </div>
            </div>
            <p class="text-muted mb-0" style="font-size:0.875rem;">
              Disease-specific dietary guidance for Diabetes, Hypertension, PCOS, Heart Disease, and more.
            </p>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card-custom p-4" onclick="showPage('analyzer', document.querySelectorAll('.nav-link-custom')[4])" style="cursor:pointer;">
            <div class="d-flex align-items-center gap-3 mb-3">
              <div class="agent-icon-box icon-purple"><i class="bi bi-clipboard2-pulse"></i></div>
              <div>
                <h6 class="fw-600 mb-0">Meal Analysis Agent</h6>
                <small class="text-muted">Agent 4</small>
              </div>
            </div>
            <p class="text-muted mb-0" style="font-size:0.875rem;">
              Describe your meals and get an AI-powered nutritional analysis, deficiency check, and improvement tips.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== PAGE: NUTRITION CHAT ===== -->
    <div id="page-chat" class="page">
      <div class="card-custom p-4" style="max-width:800px;">
        <div class="d-flex align-items-center gap-3 mb-4">
          <div class="agent-icon-box icon-green"><i class="bi bi-chat-quote"></i></div>
          <div>
            <h5 class="fw-600 mb-0">Nutrition Knowledge Agent</h5>
            <small class="text-muted">Ask any nutrition-related question · Agent 1 · IBM Granite Powered</small>
          </div>
        </div>

        <!-- Chat Messages -->
        <div class="chat-messages mb-3" id="chat-messages">
          <div class="chat-msg">
            <div class="chat-avatar avatar-ai">🥗</div>
            <div class="chat-bubble ai-bubble">
              Hello! I'm your NutriWise Nutrition Knowledge Agent, powered by IBM watsonx.ai Granite. 🌿<br><br>
              Ask me anything about nutrition — food benefits, vitamins, protein sources, healthy eating tips, and more!
            </div>
          </div>
        </div>

        <!-- Example Questions -->
        <div class="mb-3">
          <small class="text-muted fw-500">💡 Try asking:</small>
          <div class="d-flex flex-wrap gap-2 mt-2">
            <button class="btn btn-sm" style="background:#e8f5f0;color:#1a7f5a;border:1px solid #b8dece;border-radius:20px;font-size:0.78rem;"
              onclick="fillQuestion(this)">What are the benefits of oats?</button>
            <button class="btn btn-sm" style="background:#e8f5f0;color:#1a7f5a;border:1px solid #b8dece;border-radius:20px;font-size:0.78rem;"
              onclick="fillQuestion(this)">Which foods are rich in protein?</button>
            <button class="btn btn-sm" style="background:#e8f5f0;color:#1a7f5a;border:1px solid #b8dece;border-radius:20px;font-size:0.78rem;"
              onclick="fillQuestion(this)">Is paneer healthy for muscle gain?</button>
            <button class="btn btn-sm" style="background:#e8f5f0;color:#1a7f5a;border:1px solid #b8dece;border-radius:20px;font-size:0.78rem;"
              onclick="fillQuestion(this)">What foods contain Vitamin B12?</button>
          </div>
        </div>

        <!-- Input -->
        <div class="d-flex gap-2">
          <input type="text" id="chat-input" class="form-control"
            placeholder="Type your nutrition question here..."
            onkeydown="if(event.key==='Enter') sendChat()"/>
          <button class="btn-primary-custom d-flex align-items-center gap-2" onclick="sendChat()" id="chat-btn">
            <i class="bi bi-send"></i> Ask
          </button>
        </div>
      </div>
    </div>

    <!-- ===== PAGE: DIET PLANNER ===== -->
    <div id="page-planner" class="page">
      <div class="row g-4" style="max-width:1000px;">
        <div class="col-lg-5">
          <div class="card-custom p-4">
            <div class="d-flex align-items-center gap-3 mb-4">
              <div class="agent-icon-box icon-blue"><i class="bi bi-calendar-check"></i></div>
              <div>
                <h5 class="fw-600 mb-0">Diet Planner Agent</h5>
                <small class="text-muted">Agent 2 · IBM Granite Powered</small>
              </div>
            </div>
            <div class="row g-3">
              <div class="col-6">
                <label class="form-label fw-500 small">Age (years)</label>
                <input type="number" id="age" class="form-control" value="25" min="10" max="100"/>
              </div>
              <div class="col-6">
                <label class="form-label fw-500 small">Gender</label>
                <select id="gender" class="form-select">
                  <option>Male</option><option>Female</option><option>Other</option>
                </select>
              </div>
              <div class="col-6">
                <label class="form-label fw-500 small">Height (cm)</label>
                <input type="number" id="height" class="form-control" value="170" min="100" max="250"/>
              </div>
              <div class="col-6">
                <label class="form-label fw-500 small">Weight (kg)</label>
                <input type="number" id="weight" class="form-control" value="70" min="30" max="300"/>
              </div>
              <div class="col-12">
                <label class="form-label fw-500 small">Dietary Preference</label>
                <select id="dietary_pref" class="form-select">
                  <option>Vegetarian</option>
                  <option>Vegan</option>
                  <option>Non-Vegetarian</option>
                  <option>Eggetarian</option>
                  <option>Pescatarian</option>
                </select>
              </div>
              <div class="col-12">
                <label class="form-label fw-500 small">Activity Level</label>
                <select id="activity_level" class="form-select">
                  <option>Sedentary (little or no exercise)</option>
                  <option>Lightly Active (1-3 days/week)</option>
                  <option selected>Moderately Active (3-5 days/week)</option>
                  <option>Very Active (6-7 days/week)</option>
                  <option>Extra Active (athlete/physical job)</option>
                </select>
              </div>
              <div class="col-12">
                <label class="form-label fw-500 small">Fitness Goal</label>
                <select id="fitness_goal" class="form-select">
                  <option>Weight Loss</option>
                  <option>Weight Gain</option>
                  <option>Muscle Gain</option>
                  <option selected>General Wellness</option>
                  <option>Improved Energy</option>
                  <option>Better Digestion</option>
                </select>
              </div>
              <div class="col-12">
                <button class="btn-primary-custom w-100" onclick="generateMealPlan()" id="plan-btn">
                  <i class="bi bi-magic me-2"></i>Generate My Meal Plan
                </button>
              </div>
            </div>
          </div>
        </div>
        <div class="col-lg-7">
          <div class="card-custom p-4" style="min-height:200px;">
            <h6 class="fw-600 mb-3"><i class="bi bi-journal-text me-2 text-primary"></i>Your Personalized Meal Plan</h6>
            <div class="response-box" id="plan-response">
              <div class="loading-spinner d-none" id="plan-loader">
                <div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div>
                <span>IBM Granite is crafting your meal plan…</span>
              </div>
              <div class="response-content" id="plan-content"></div>
            </div>
            <div id="plan-placeholder" class="text-center text-muted py-5">
              <i class="bi bi-calendar3" style="font-size:2.5rem;opacity:0.3;"></i>
              <p class="mt-3 small">Fill in your profile and click "Generate My Meal Plan"</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== PAGE: HEALTH ADVISOR ===== -->
    <div id="page-advisor" class="page">
      <div class="row g-4" style="max-width:1000px;">
        <div class="col-lg-5">
          <div class="card-custom p-4">
            <div class="d-flex align-items-center gap-3 mb-4">
              <div class="agent-icon-box icon-orange"><i class="bi bi-heart-pulse"></i></div>
              <div>
                <h5 class="fw-600 mb-0">Health Advisory Agent</h5>
                <small class="text-muted">Agent 3 · IBM Granite Powered</small>
              </div>
            </div>
            <p class="text-muted small mb-3">Select one or more health conditions to receive personalized dietary and lifestyle guidance:</p>
            <div class="d-flex flex-column gap-2" id="conditions-list">
              <label class="condition-check"><input type="checkbox" value="Diabetes" class="form-check-input me-1"/> 🍬 Diabetes</label>
              <label class="condition-check"><input type="checkbox" value="Hypertension" class="form-check-input me-1"/> 💓 Hypertension (High Blood Pressure)</label>
              <label class="condition-check"><input type="checkbox" value="Obesity" class="form-check-input me-1"/> ⚖️ Obesity</label>
              <label class="condition-check"><input type="checkbox" value="Heart Disease" class="form-check-input me-1"/> ❤️ Heart Disease</label>
              <label class="condition-check"><input type="checkbox" value="PCOS" class="form-check-input me-1"/> 🌸 PCOS (Polycystic Ovary Syndrome)</label>
              <label class="condition-check"><input type="checkbox" value="High Cholesterol" class="form-check-input me-1"/> 🔬 High Cholesterol</label>
            </div>
            <button class="btn-primary-custom w-100 mt-4" onclick="getHealthAdvice()" id="advisor-btn">
              <i class="bi bi-shield-heart me-2"></i>Get Health Advisory
            </button>
          </div>
        </div>
        <div class="col-lg-7">
          <div class="card-custom p-4" style="min-height:200px;">
            <h6 class="fw-600 mb-3"><i class="bi bi-clipboard2-heart me-2" style="color:#d97706;"></i>Health Advisory Report</h6>
            <div class="response-box" id="advisor-response">
              <div class="loading-spinner d-none" id="advisor-loader">
                <div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div>
                <span>IBM Granite is generating your advisory…</span>
              </div>
              <div class="response-content" id="advisor-content"></div>
            </div>
            <div id="advisor-placeholder" class="text-center text-muted py-5">
              <i class="bi bi-heart-pulse" style="font-size:2.5rem;opacity:0.3;color:#d97706;"></i>
              <p class="mt-3 small">Select health conditions and click "Get Health Advisory"</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== PAGE: MEAL ANALYZER ===== -->
    <div id="page-analyzer" class="page">
      <div class="row g-4" style="max-width:1000px;">
        <div class="col-lg-5">
          <div class="card-custom p-4">
            <div class="d-flex align-items-center gap-3 mb-4">
              <div class="agent-icon-box icon-purple"><i class="bi bi-clipboard2-pulse"></i></div>
              <div>
                <h5 class="fw-600 mb-0">Meal Analysis Agent</h5>
                <small class="text-muted">Agent 4 · IBM Granite Powered</small>
              </div>
            </div>
            <label class="form-label fw-500 small">Describe your meals for the day:</label>
            <textarea id="meal-input" class="form-control mb-2" rows="10"
              placeholder="Example:
Breakfast:
2 Rotis with butter
1 cup of chai

Lunch:
Rice (1 bowl)
Paneer curry
Dal

Snack:
Biscuits and tea

Dinner:
Vegetable salad
1 cup curd"></textarea>
            <button class="btn-primary-custom w-100" onclick="analyzeMeal()" id="analyzer-btn">
              <i class="bi bi-search me-2"></i>Analyze My Meals
            </button>
          </div>
        </div>
        <div class="col-lg-7">
          <div class="card-custom p-4" style="min-height:200px;">
            <h6 class="fw-600 mb-3"><i class="bi bi-bar-chart me-2" style="color:#7c3aed;"></i>Meal Analysis Report</h6>
            <div class="response-box" id="analyzer-response">
              <div class="loading-spinner d-none" id="analyzer-loader">
                <div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div>
                <span>IBM Granite is analyzing your meals…</span>
              </div>
              <div class="response-content" id="analyzer-content"></div>
            </div>
            <div id="analyzer-placeholder" class="text-center text-muted py-5">
              <i class="bi bi-clipboard2-pulse" style="font-size:2.5rem;opacity:0.3;color:#7c3aed;"></i>
              <p class="mt-3 small">Enter your meals and click "Analyze My Meals"</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== PAGE: ABOUT ===== -->
    <div id="page-about" class="page" style="max-width:860px;">
      <div class="hero-gradient mb-4 p-4">
        <h4 class="fw-700">About NutriWise AI</h4>
        <p class="mb-0" style="opacity:0.85;font-size:0.9rem;">
          A multi-agent AI nutrition assistant built on IBM watsonx.ai Granite Models — demonstrating agentic AI in the healthcare and wellness domain.
        </p>
      </div>

      <div class="card-custom p-4 mb-4">
        <h6 class="fw-600 mb-3">🤖 Multi-Agent Architecture</h6>
        <div class="arch-step">
          <div class="d-flex align-items-center gap-2 mb-1">
            <span class="badge bg-success">Agent 1</span>
            <strong>Nutrition Knowledge Agent</strong>
          </div>
          <p class="text-muted small mb-0">Answers any nutrition question using IBM Granite. Routes user queries into structured educational responses about food, nutrients, and dietary science.</p>
        </div>
        <div class="arch-step" style="border-left-color:#2563eb;">
          <div class="d-flex align-items-center gap-2 mb-1">
            <span class="badge bg-primary">Agent 2</span>
            <strong>Diet Planner Agent</strong>
          </div>
          <p class="text-muted small mb-0">Generates personalized daily meal plans based on age, weight, height, activity level, dietary preference, and fitness goals using IBM Granite.</p>
        </div>
        <div class="arch-step" style="border-left-color:#d97706;">
          <div class="d-flex align-items-center gap-2 mb-1">
            <span class="badge bg-warning text-dark">Agent 3</span>
            <strong>Health Advisory Agent</strong>
          </div>
          <p class="text-muted small mb-0">Provides disease-specific dietary and lifestyle recommendations for Diabetes, Hypertension, PCOS, Heart Disease, Obesity, and Cholesterol using IBM Granite.</p>
        </div>
        <div class="arch-step" style="border-left-color:#7c3aed;">
          <div class="d-flex align-items-center gap-2 mb-1">
            <span class="badge bg-purple" style="background:#7c3aed;">Agent 4</span>
            <strong>Meal Analysis Agent</strong>
          </div>
          <p class="text-muted small mb-0">Analyzes user-submitted meals in free-text format and uses IBM Granite to estimate nutritional quality, identify deficiencies, and suggest improvements.</p>
        </div>
      </div>

      <div class="card-custom p-4 mb-4">
        <h6 class="fw-600 mb-3">⚡ Orchestrator Pattern</h6>
        <p class="text-muted small">A central orchestrator function routes each user request to the appropriate specialized agent. All agents share a single <code>generate_response()</code> function that calls the IBM watsonx.ai Granite inference endpoint — ensuring consistency, reusability, and clean separation of concerns.</p>
        <div class="p-3 mt-2" style="background:#f4f8f6;border-radius:10px;font-family:monospace;font-size:0.8rem;color:#1a7f5a;">
          User Request → Orchestrator → [Agent 1 | Agent 2 | Agent 3 | Agent 4]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generate_response(prompt) → IBM watsonx.ai Granite → AI Response
        </div>
      </div>

      <div class="card-custom p-4">
        <h6 class="fw-600 mb-3">🛠️ Technology Stack</h6>
        <div>
          <span class="tech-pill">IBM watsonx.ai</span>
          <span class="tech-pill">IBM Granite-13B-Instruct</span>
          <span class="tech-pill">Python 3.10+</span>
          <span class="tech-pill">Flask</span>
          <span class="tech-pill">Bootstrap 5</span>
          <span class="tech-pill">JavaScript (Fetch API)</span>
          <span class="tech-pill">IBM IAM Authentication</span>
          <span class="tech-pill">Multi-Agent Architecture</span>
          <span class="tech-pill">Agentic AI</span>
          <span class="tech-pill">REST API</span>
        </div>
      </div>
    </div>

  </div><!-- /page-content -->
</div><!-- /main-content -->

<!-- ========== JAVASCRIPT ========== -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>

// ===========================================================
// UTILITY FUNCTIONS
// ===========================================================

function escapeHtml(t) {
  return t.replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
}

// Converts simple markdown (## headings, **bold**, - lists) to HTML.
// Uses string operations only — no regex with / in replacement strings.
function renderContent(text) {
  var lines = text.split('\\n');
  var out = [];
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    // ## Heading
    if (line.match(/^## /)) {
      line = '<h6 style="font-weight:700;color:#1a7f5a;margin-top:14px;margin-bottom:4px;">' +
             line.replace(/^## /, '') + '</h6>';
    }
    // **bold**
    line = line.replace(/[*][*]([^*]+)[*][*]/g, '<strong>$1</strong>');
    // - list item
    if (line.match(/^- /)) {
      line = '• ' + line.replace(/^- /, '');
    }
    out.push(line);
  }
  return out.join('<br>');
}

// ===========================================================
// NAVIGATION
// ===========================================================

function showPage(name, el) {
  document.querySelectorAll('.page').forEach(function(p) {
    p.classList.remove('active');
  });
  var target = document.getElementById('page-' + name);
  if (target) target.classList.add('active');

  document.querySelectorAll('.nav-link-custom').forEach(function(a) {
    a.classList.remove('active');
  });
  if (el) el.classList.add('active');

  var titles = {
    'home':     'Welcome to NutriWise AI',
    'chat':     'Nutrition Knowledge Agent',
    'planner':  'Diet Planner Agent',
    'advisor':  'Health Advisory Agent',
    'analyzer': 'Meal Analysis Agent',
    'about':    'About NutriWise AI'
  };
  var titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = titles[name] || 'NutriWise AI';

  if (window.innerWidth < 768) {
    document.getElementById('sidebar').classList.remove('open');
  }
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ===========================================================
// AGENT 1 — Nutrition Chat
// ===========================================================

function fillQuestion(btn) {
  document.getElementById('chat-input').value = btn.textContent.trim();
  document.getElementById('chat-input').focus();
}

function appendMessage(html, isUser) {
  var msgs = document.getElementById('chat-messages');
  var div = document.createElement('div');
  div.className = 'chat-msg' + (isUser ? ' user' : '');
  var avatar = document.createElement('div');
  avatar.className = 'chat-avatar ' + (isUser ? 'avatar-user' : 'avatar-ai');
  avatar.innerHTML = isUser ? '<i class="bi bi-person"></i>' : '🥗';
  var bubble = document.createElement('div');
  bubble.className = 'chat-bubble ' + (isUser ? 'user-bubble' : 'ai-bubble');
  bubble.innerHTML = html;
  div.appendChild(avatar);
  div.appendChild(bubble);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

async function sendChat() {
  var input = document.getElementById('chat-input');
  var question = input.value.trim();
  if (!question) return;

  appendMessage(escapeHtml(question), true);

  // Typing indicator
  var typingDiv = appendMessage(
    '<div class="loading-spinner"><div class="spinner-dot"></div>' +
    '<div class="spinner-dot"></div><div class="spinner-dot"></div>' +
    '<span style="font-size:0.8rem;">IBM Granite is thinking...</span></div>',
    false
  );

  input.value = '';
  var btn = document.getElementById('chat-btn');
  btn.disabled = true;

  try {
    // IBM watsonx.ai call via Flask → Agent 1
    var res = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: 'nutrition_knowledge', question: question })
    });
    var data = await res.json();
    typingDiv.remove();
    appendMessage(renderContent(data.response || data.error || 'No response.'), false);
  } catch (e) {
    typingDiv.remove();
    appendMessage('<span style="color:red;">Error connecting to IBM watsonx.ai. Check your credentials.</span>', false);
  }
  btn.disabled = false;
}

// ===========================================================
// AGENT 2 — Diet Planner
// ===========================================================

async function generateMealPlan() {
  var btn = document.getElementById('plan-btn');
  btn.disabled = true;
  var resp        = document.getElementById('plan-response');
  var loader      = document.getElementById('plan-loader');
  var content     = document.getElementById('plan-content');
  var placeholder = document.getElementById('plan-placeholder');

  placeholder.style.display = 'none';
  resp.classList.add('visible');
  loader.classList.remove('d-none');
  content.innerHTML = '';

  try {
    // IBM watsonx.ai call via Flask → Agent 2
    var res = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent:          'diet_planner',
        age:            document.getElementById('age').value,
        gender:         document.getElementById('gender').value,
        height:         document.getElementById('height').value,
        weight:         document.getElementById('weight').value,
        dietary_pref:   document.getElementById('dietary_pref').value,
        activity_level: document.getElementById('activity_level').value,
        fitness_goal:   document.getElementById('fitness_goal').value
      })
    });
    var data = await res.json();
    loader.classList.add('d-none');
    content.innerHTML = renderContent(data.response || data.error || 'No response.');
  } catch (e) {
    loader.classList.add('d-none');
    content.innerHTML = '<span style="color:red;">Error connecting to IBM watsonx.ai.</span>';
  }
  btn.disabled = false;
}

// ===========================================================
// AGENT 3 — Health Advisor
// ===========================================================

async function getHealthAdvice() {
  var boxes   = document.querySelectorAll('#conditions-list input:checked');
  var checked = Array.prototype.slice.call(boxes).map(function(c) { return c.value; });
  if (checked.length === 0) {
    alert('Please select at least one health condition.');
    return;
  }
  var btn         = document.getElementById('advisor-btn');
  btn.disabled    = true;
  var resp        = document.getElementById('advisor-response');
  var loader      = document.getElementById('advisor-loader');
  var content     = document.getElementById('advisor-content');
  var placeholder = document.getElementById('advisor-placeholder');

  placeholder.style.display = 'none';
  resp.classList.add('visible');
  loader.classList.remove('d-none');
  content.innerHTML = '';

  try {
    // IBM watsonx.ai call via Flask → Agent 3
    var res = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: 'health_advisory', conditions: checked })
    });
    var data = await res.json();
    loader.classList.add('d-none');
    content.innerHTML = renderContent(data.response || data.error || 'No response.');
  } catch (e) {
    loader.classList.add('d-none');
    content.innerHTML = '<span style="color:red;">Error connecting to IBM watsonx.ai.</span>';
  }
  btn.disabled = false;
}

// ===========================================================
// AGENT 4 — Meal Analyzer
// ===========================================================

async function analyzeMeal() {
  var mealText = document.getElementById('meal-input').value.trim();
  if (!mealText) {
    alert('Please describe your meals first.');
    return;
  }
  var btn         = document.getElementById('analyzer-btn');
  btn.disabled    = true;
  var resp        = document.getElementById('analyzer-response');
  var loader      = document.getElementById('analyzer-loader');
  var content     = document.getElementById('analyzer-content');
  var placeholder = document.getElementById('analyzer-placeholder');

  placeholder.style.display = 'none';
  resp.classList.add('visible');
  loader.classList.remove('d-none');
  content.innerHTML = '';

  try {
    // IBM watsonx.ai call via Flask → Agent 4
    var res = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: 'meal_analysis', meal_text: mealText })
    });
    var data = await res.json();
    loader.classList.add('d-none');
    content.innerHTML = renderContent(data.response || data.error || 'No response.');
  } catch (e) {
    loader.classList.add('d-none');
    content.innerHTML = '<span style="color:red;">Error connecting to IBM watsonx.ai.</span>';
  }
  btn.disabled = false;
}

// ===========================================================
// ENTER KEY — Chat input
// ===========================================================
document.addEventListener('DOMContentLoaded', function() {
  var chatInput = document.getElementById('chat-input');
  if (chatInput) {
    chatInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') sendChat();
    });
  }
});
</script>
</body>
</html>"""


# =============================================================================
# Flask Routes
# =============================================================================

@app.route("/")
def home():
    """
    Render the main NutriWise AI single-page application.
    We return the HTML directly (bypassing Jinja2 template rendering) because
    the embedded JavaScript uses ${...} template literals that Jinja2 would
    incorrectly try to evaluate as template variables.
    """
    return app.response_class(BASE_TEMPLATE, mimetype="text/html")


@app.route("/api/agent", methods=["POST"])
def agent_api():
    """
    Central API endpoint that receives agent requests from the frontend.
    Passes the request to the orchestrator which routes to the correct agent.
    All agents internally call IBM watsonx.ai Granite via generate_response().
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request payload."}), 400

    agent_name = data.get("agent", "")

    try:
        # Orchestrator routes to the appropriate AI agent
        result = orchestrate(agent_name, data)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI – Personalized Nutrition Coach")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print(f"  API Key   : {'✅ Set' if WATSONX_API_KEY else '❌ NOT SET — check .env'}")
    print(f"  Project ID: {'✅ Set' if WATSONX_PROJECT_ID else '❌ NOT SET — check .env'}")
    print(f"  URL       : {WATSONX_URL}")
    print(f"  Model ID  : {WATSONX_MODEL_ID}")
    print("=" * 60)
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
