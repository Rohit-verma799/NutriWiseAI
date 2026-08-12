<<<<<<< HEAD
# NutriWiseAI
=======
# 🥗 NutriWise AI – Personalized Nutrition Coach

> **Powered by IBM watsonx.ai Granite Foundation Models**  
> A multi-agent AI web application providing personalized diet planning, interactive nutrition counseling, health advisory, and real-time meal macro/calorie analysis.

---

## 🌟 Overview

**NutriWise AI** leverages IBM watsonx.ai foundation models (such as `ibm/granite-4-h-small`) within a multi-agent framework to deliver tailored, science-backed nutrition guidance. Whether you are looking to lose weight, gain muscle, manage medical conditions, or analyze daily meal macros, NutriWise AI acts as your 24/7 personal nutrition assistant.

---

## 🤖 Multi-Agent Architecture

NutriWise AI consists of 4 specialized AI agents managed by a central **Orchestrator**:

1. 📚 **Nutrition Knowledge Agent**  
   Answers general nutrition queries, debunks dietary myths, and explains macronutrients/micronutrients.
2. 🗓️ **Personalized Diet Planner Agent**  
   Generates customized 7-day meal plans tailored to age, gender, height, weight, activity level, dietary preferences (e.g. Vegan, Keto), and fitness goals.
3. 🩺 **Health Advisory & Condition Management Agent**  
   Provides tailored dietary advice and precautions for managing health conditions such as Diabetes, Hypertension, PCOS, Celiac, and High Cholesterol.
4. 🍽️ **Meal Calorie & Macro Analysis Agent**  
   Breaks down user-submitted meals into estimated calories, protein, carbohydrates, fats, and micronutrient profiles.

---

## 🚀 Technology Stack

- **Backend Framework**: Python, Flask
- **AI Core**: IBM watsonx.ai (Granite 4H Small, Llama 3.3 70B Instruct, Mistral Small 3.1)
- **Authentication**: IBM Cloud IAM OAuth 2.0 Token Generation
- **Frontend**: Modern Responsive HTML5 / CSS3 / JavaScript Single Page Application (SPA)
- **Environment Management**: `python-dotenv`

---

## 📁 Repository Structure

```text
NutriWise/
├── app.py               # Main Flask application, API endpoints & Multi-Agent Orchestrator
├── requirements.txt     # Python dependency list (Flask, python-dotenv, requests)
├── .env.example         # Environment variable template with placeholders
├── .gitignore           # Git ignore file ensuring secrets (.env) are not committed
└── README.md            # Project documentation
```

---

## ⚙️ Prerequisites

- **Python 3.9+** installed on your system.
- An **IBM Cloud Account** with active access to **IBM watsonx.ai**.
- IBM Cloud API Key and watsonx.ai Project ID.

---

## 🛠️ Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd NutriWise
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```
Open `.env` and fill in your actual IBM Cloud watsonx.ai credentials:
```env
WATSONX_API_KEY=your_actual_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_actual_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-4-h-small
```

---

## 🏃 Running the Application

Start the local Flask server:
```bash
python app.py
```

Once started, open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔌 API Endpoints

### `POST /api/agent`
Central endpoint for invoking NutriWise AI agents.

**Request Payload Examples:**

- **Nutrition Knowledge Query:**
  ```json
  {
    "agent": "nutrition_knowledge",
    "question": "What are the benefits of intermittent fasting?"
  }
  ```

- **Personalized Diet Plan:**
  ```json
  {
    "agent": "diet_planner",
    "age": 28,
    "gender": "Female",
    "height": 165,
    "weight": 60,
    "dietary_pref": "Vegetarian",
    "activity_level": "Moderate",
    "fitness_goal": "Weight Maintenance"
  }
  ```

- **Health Advisory:**
  ```json
  {
    "agent": "health_advisory",
    "conditions": ["Diabetes Type 2", "Hypertension"]
  }
  ```

- **Meal Analysis:**
  ```json
  {
    "agent": "meal_analysis",
    "meal_text": "2 scrambled eggs, 1 slice of whole wheat toast, and 1 black coffee"
  }
  ```

---

## 🔒 Security Best Practices

- **Never commit `.env` to version control.** Secrets must remain local.
- `.env.example` is provided as a template containing placeholders only.
- The `.gitignore` file is configured to exclude sensitive configuration files.

---

## 📜 License

This project is developed for educational and submission purposes under the **IBM watsonx.ai Challenge**.
>>>>>>> master
