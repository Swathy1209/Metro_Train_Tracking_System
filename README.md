# 🚇 Metro Navigation System

An interactive metro travel assistant and chatbot built using Streamlit, NetworkX, and Folium. Plan routes, explore maps, analyze traffic, and interact with a smart assistant.

## 🚀 Features

- Smart route planner with time and cost optimization  
- Interactive metro map with Folium and traffic heatmaps  
- Station finder by GPS or landmark  
- Real-time chatbot for metro queries  
- Traffic and network analysis dashboard  
- Voice input integration  

## 🧰 Tech Stack

- Python  
- Streamlit  
- NetworkX  
- Folium + streamlit-folium  
- Plotly + Altair + Matplotlib  
- SpeechRecognition (for voice input)  

## 📂 Project Structure

```
.
├── metro_trainapp.py         # Main Streamlit app
├── data/                     # (Optional) Directory for future datasets
└── README.md                 # Project documentation
```

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/metro-navigation-system.git
cd metro-navigation-system
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Or manually install key packages:

```bash
pip install streamlit pandas numpy networkx folium streamlit-folium matplotlib plotly altair speechrecognition pillow
```

### 4. Run the App

```bash
streamlit run metro_trainapp.py
```

## 📌 Use Cases

- Plan the fastest or cheapest metro route  
- Visualize live traffic and station connections  
- Find the nearest station using GPS or landmarks  
- Ask questions like “Which line goes to University?”  
- Analyze passenger flow and station centrality  

## 📄 License

MIT License
