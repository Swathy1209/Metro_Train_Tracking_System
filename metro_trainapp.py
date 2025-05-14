import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import datetime
import speech_recognition as sr
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import time
import altair as alt
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Metro Navigation System",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable dark mode toggle
def set_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .main {background-color: #0E1117; color: #FAFAFA;}
        .st-bw {background-color: #262730; color: #FAFAFA;}
        .css-1aumxhk {background-color: #1E1E1E; color: #FAFAFA;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main {background-color: #FFFFFF; color: #111111;}
        .st-bw {background-color: #F0F2F6; color: #111111;}
        </style>
        """, unsafe_allow_html=True)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
    
if 'user_routes' not in st.session_state:
    st.session_state.user_routes = []
    
if 'metro_graph' not in st.session_state:
    # Initialize metro graph (this will be loaded from data later)
    st.session_state.metro_graph = nx.Graph()

# Sample metro data - in production, this would be loaded from a database or API
@st.cache_data
def load_metro_data():
    # Sample stations with coordinates (lat, long)
    stations_data = {
        "Central": {"position": (28.6315, 77.2167), "lines": ["Red", "Blue"], "avg_traffic": 850},
        "Civic Center": {"position": (28.6292, 77.2274), "lines": ["Red"], "avg_traffic": 650},
        "City Park": {"position": (28.6426, 77.2091), "lines": ["Blue"], "avg_traffic": 750},
        "Riverside": {"position": (28.6214, 77.2387), "lines": ["Blue", "Green"], "avg_traffic": 550},
        "Commerce Square": {"position": (28.6354, 77.2270), "lines": ["Green"], "avg_traffic": 450},
        "University": {"position": (28.6189, 77.2066), "lines": ["Red", "Green"], "avg_traffic": 700},
        "Tech Hub": {"position": (28.6401, 77.2194), "lines": ["Yellow"], "avg_traffic": 600},
        "Market Street": {"position": (28.6269, 77.2318), "lines": ["Yellow", "Blue"], "avg_traffic": 800},
        "North Terminal": {"position": (28.6475, 77.2118), "lines": ["Yellow", "Red"], "avg_traffic": 500},
        "South Gate": {"position": (28.6160, 77.2149), "lines": ["Green"], "avg_traffic": 350},
    }
    
    # Sample connections between stations with travel time in minutes and fare in currency units
    connections = [
        ("Central", "Civic Center", {"time": 5, "fare": 10}),
        ("Central", "City Park", {"time": 6, "fare": 12}),
        ("Central", "Tech Hub", {"time": 8, "fare": 15}),
        ("Civic Center", "Riverside", {"time": 7, "fare": 14}),
        ("Civic Center", "Commerce Square", {"time": 5, "fare": 10}),
        ("City Park", "North Terminal", {"time": 6, "fare": 12}),
        ("Riverside", "University", {"time": 9, "fare": 18}),
        ("Riverside", "Market Street", {"time": 4, "fare": 8}),
        ("Commerce Square", "Market Street", {"time": 6, "fare": 12}),
        ("University", "South Gate", {"time": 5, "fare": 10}),
        ("Tech Hub", "North Terminal", {"time": 7, "fare": 14}),
        ("Market Street", "Tech Hub", {"time": 8, "fare": 16}),
    ]
    
    # Sample landmarks near stations
    landmarks = {
        "National Museum": {"nearest_station": "Central", "distance": 0.5},
        "City Hall": {"nearest_station": "Civic Center", "distance": 0.2},
        "Central Park": {"nearest_station": "City Park", "distance": 0.1},
        "Shopping Mall": {"nearest_station": "Market Street", "distance": 0.3},
        "University Campus": {"nearest_station": "University", "distance": 0.0},
        "Business District": {"nearest_station": "Commerce Square", "distance": 0.4},
        "Technology Park": {"nearest_station": "Tech Hub", "distance": 0.2},
        "River Walk": {"nearest_station": "Riverside", "distance": 0.3},
        "Sports Stadium": {"nearest_station": "North Terminal", "distance": 0.6},
        "Botanical Garden": {"nearest_station": "South Gate", "distance": 0.4},
    }
    
    # Sample traffic data (hourly averages)
    traffic_data = {}
    for station in stations_data:
        base_traffic = stations_data[station]["avg_traffic"]
        daily_pattern = []
        for hour in range(24):
            # Simulate peak hours
            if 7 <= hour <= 9:
                multiplier = 1.5 + np.random.normal(0, 0.2)
            elif 17 <= hour <= 19:
                multiplier = 1.6 + np.random.normal(0, 0.2)
            elif 0 <= hour <= 5:
                multiplier = 0.2 + np.random.normal(0, 0.1)
            else:
                multiplier = 1.0 + np.random.normal(0, 0.15)
            
            daily_pattern.append(max(0, int(base_traffic * multiplier)))
        
        traffic_data[station] = daily_pattern
    
    # Create a metro graph
    G = nx.Graph()
    
    # Add nodes (stations)
    for station, data in stations_data.items():
        G.add_node(station, 
                  position=data["position"], 
                  lines=data["lines"],
                  avg_traffic=data["avg_traffic"])
    
    # Add edges (connections)
    G.add_edges_from(connections)
    
    return {
        "stations": stations_data,
        "connections": connections,
        "landmarks": landmarks,
        "traffic_data": traffic_data,
        "graph": G
    }
# Ensure all station names and lines are strings to avoid 'float' split errors
    for station in stations_data:
        station = str(station)
        stations_data[station]["lines"] = [str(line) for line in stations_data[station]["lines"]]

    G = nx.Graph()
    for station, data in stations_data.items():
        G.add_node(station, position=data["position"], lines=data["lines"], avg_traffic=data["avg_traffic"])
    G.add_edges_from(connections)

    return {"stations": stations_data, "connections": connections, "graph": G}

# Load data
metro_data = load_metro_data()
st.session_state.metro_graph = metro_data["graph"]

# Function to find shortest path
def find_shortest_path(graph, start, end, metric="time"):
    try:
        path = nx.shortest_path(graph, start, end, weight=metric)
        
        # Calculate total time and fare
        total_time = 0
        total_fare = 0
        
        for i in range(len(path) - 1):
            edge_data = graph[path[i]][path[i+1]]
            total_time += edge_data["time"]
            total_fare += edge_data["fare"]
            
        return {
            "path": path,
            "total_time": total_time,
            "total_fare": total_fare
        }
    except nx.NetworkXNoPath:
        return None

# Function to find multiple alternate routes
def find_multiple_routes(graph, start, end, num_routes=3):
    routes = []
    
    # First route is the shortest path by time
    shortest_route = find_shortest_path(graph, start, end, "time")
    if shortest_route:
        routes.append(shortest_route)
        
        # Create a temporary graph to manipulate
        temp_graph = graph.copy()
        
        # Try to find alternative routes by temporarily removing edges
        for _ in range(num_routes - 1):
            if len(routes) > 0:
                # Remove a random edge from the previous route
                prev_path = routes[-1]["path"]
                if len(prev_path) > 2:  # Make sure we have edges to remove
                    edge_to_remove = (prev_path[1], prev_path[2])  # Remove the second edge
                    if temp_graph.has_edge(*edge_to_remove):
                        temp_graph.remove_edge(*edge_to_remove)
                        
                # Try to find a new path
                try:
                    alt_path = find_shortest_path(temp_graph, start, end, "time")
                    if alt_path and alt_path["path"] not in [r["path"] for r in routes]:
                        routes.append(alt_path)
                except:
                    pass
    
    return routes

# Function to calculate fare based on distance and time
def calculate_fare(route):
    base_fare = 10
    distance_factor = len(route["path"]) - 1  # Number of edges
    time_factor = route["total_time"] / 10
    
    return base_fare + (distance_factor * 5) + (time_factor * 2)

# Function to find routes based on time constraints
def find_time_based_routes(graph, start, end, depart_time):
    # Convert departure time to hour of day
    hour = depart_time.hour
    
    # Create a temporary graph with adjusted weights
    temp_graph = graph.copy()
    
    # Adjust travel times based on typical traffic patterns
    for u, v, data in temp_graph.edges(data=True):
        # Peak hours: increase travel time
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            temp_graph[u][v]['time'] = data['time'] * 1.3
        # Late night: reduce travel time
        elif 22 <= hour or hour <= 5:
            temp_graph[u][v]['time'] = data['time'] * 0.9
    
    # Find routes in the adjusted graph
    return find_multiple_routes(temp_graph, start, end)

# Function to find nearest station to a landmark
def find_station_by_landmark(landmark):
    for lm, data in metro_data["landmarks"].items():
        if landmark.lower() in lm.lower():
            return data["nearest_station"]
    return None

def find_nearest_station(user_lat, user_lng):
    nearest = None
    min_dist = float('inf')
    
    for station, data in metro_data["stations"].items():
        lat, lng = data["position"]
        # Simple distance calculation (not actual walking distance)
        dist = ((user_lat - lat) ** 2 + (user_lng - lng) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = station
            
    return nearest, min_dist * 111  # Approximate conversion to km

# Function to create line-filtered graph
def filter_by_lines(lines):
    filtered_graph = nx.Graph()
    
    # Add nodes that have at least one of the selected lines
    for node, data in metro_data["graph"].nodes(data=True):
        node_lines = data.get("lines", [])
        if any(line in node_lines for line in lines):
            filtered_graph.add_node(node, **data)
    
    # Add edges between nodes that are in the filtered graph
    for u, v, data in metro_data["graph"].edges(data=True):
        if filtered_graph.has_node(u) and filtered_graph.has_node(v):
            filtered_graph.add_edge(u, v, **data)
    
    return filtered_graph

# Function to create a folium map with the metro network
def create_metro_map(highlighted_path=None, traffic_heatmap=False, current_hour=None):
    # Create a folium map centered at the average position of all stations
    positions = [data["position"] for data in metro_data["stations"].values()]
    center_lat = sum(pos[0] for pos in positions) / len(positions)
    center_lng = sum(pos[1] for pos in positions) / len(positions)
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    
    # Line colors
    line_colors = {
        "Red": "red",
        "Blue": "blue",
        "Green": "green",
        "Yellow": "orange"
    }
    
    # Add edges (connections)
    for u, v, data in metro_data["graph"].edges(data=True):
        # Get node positions
        pos_u = metro_data["graph"].nodes[u]["position"]
        pos_v = metro_data["graph"].nodes[v]["position"]
        
        # Determine line color
        common_lines = set(metro_data["graph"].nodes[u]["lines"]) & set(metro_data["graph"].nodes[v]["lines"])
        if common_lines:
            line = list(common_lines)[0]
            color = line_colors.get(line, "gray")
        else:
            color = "gray"
        
        # Check if this edge is part of the highlighted path
        if highlighted_path and u in highlighted_path and v in highlighted_path:
            # Check if they are consecutive in the path
            u_idx = highlighted_path.index(u)
            if u_idx < len(highlighted_path) - 1 and highlighted_path[u_idx + 1] == v:
                color = "purple"
                weight = 5
            elif u_idx > 0 and highlighted_path[u_idx - 1] == v:
                color = "purple"
                weight = 5
            else:
                weight = 2
        else:
            weight = 2
        
        # Draw the connection line
        folium.PolyLine(
            locations=[pos_u, pos_v],
            color=color,
            weight=weight,
            opacity=0.7
        ).add_to(m)
    
    # Add traffic heatmap if enabled
    if traffic_heatmap and current_hour is not None:
        heatmap_data = []
        for station, data in metro_data["stations"].items():
            position = data["position"]
            # Get traffic for the current hour (index is hour number)
            traffic = metro_data["traffic_data"][station][current_hour]
            # Add weighted point to heatmap
            heatmap_data.append([position[0], position[1], traffic/100])
        
        # Add heatmap layer
        folium.plugins.HeatMap(
            heatmap_data,
            radius=15,
            gradient={0.4: 'blue', 0.65: 'yellow', 0.8: 'orange', 1: 'red'}
        ).add_to(m)
    
    # Add station markers
    for station, data in metro_data["stations"].items():
        position = data["position"]
        lines = data["lines"]
        
        # Create line information text
        line_text = ", ".join([f"<span style='color:{line_colors.get(line, 'gray')}'>{line}</span>" for line in lines])
        
        # Create# Create popup content
        popup_content = f"""
        <div style='min-width: 180px'>
            <b>{station}</b><br>
            Lines: {line_text}<br>
            Average Traffic: {data['avg_traffic']} passengers/hour
        </div>
        """
        
        # Create custom icon
        icon = folium.Icon(
            icon="subway",
            prefix="fa",
            color="cadetblue" if highlighted_path and station in highlighted_path else "blue"
        )
        
        # Add marker
        folium.Marker(
            location=position,
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=station,
            icon=icon
        ).add_to(m)
    
    return m

# Function for analyzing trends
def analyze_trends():
    # Prepare data for graphs
    hourly_data = []
    for station, traffic in metro_data["traffic_data"].items():
        for hour, count in enumerate(traffic):
            hourly_data.append({
                "Station": station,
                "Hour": hour,
                "Traffic": count
            })
    
    df_traffic = pd.DataFrame(hourly_data)
    
    # Calculate average traffic by hour across all stations
    avg_by_hour = df_traffic.groupby("Hour")["Traffic"].mean().reset_index()
    
    # Calculate busiest stations
    busiest_stations = df_traffic.groupby("Station")["Traffic"].mean().sort_values(ascending=False).reset_index()
    
    return avg_by_hour, busiest_stations

# Function for graph centrality analysis
def analyze_centrality():
    G = metro_data["graph"]
    
    # Calculate different centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Create dataframe for the results
    centrality_df = pd.DataFrame({
        'Station': list(degree_centrality.keys()),
        'Degree Centrality': list(degree_centrality.values()),
        'Betweenness Centrality': list(betweenness_centrality.values()),
        'Closeness Centrality': list(closeness_centrality.values())
    })
    
    return centrality_df

# Predictive suggestions based on time and historical patterns
def get_predictive_suggestions(hour_of_day):
    # Simple rule-based suggestions
    if 7 <= hour_of_day <= 9:
        return {
            "from_stations": ["Residential Areas", "University", "South Gate"],
            "to_stations": ["Central", "Commerce Square", "Tech Hub"],
            "message": "Morning commute patterns detected. Consider avoiding Central and Tech Hub stations if possible."
        }
    elif 17 <= hour_of_day <= 19:
        return {
            "from_stations": ["Commerce Square", "Central", "Tech Hub"],
            "to_stations": ["University", "South Gate", "Riverside"],
            "message": "Evening rush hour detected. The Red Line is experiencing higher than normal traffic."
        }
    elif 11 <= hour_of_day <= 14:
        return {
            "from_stations": ["Central", "Market Street"],
            "to_stations": ["Riverside", "City Park"],
            "message": "Lunchtime travel is moderate. Good time for leisure travel."
        }
    else:
        return {
            "from_stations": [],
            "to_stations": [],
            "message": "Current travel conditions are normal across the network."
        }

# Function to handle voice input for stations
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak the station name")
        audio = r.listen(source)
        
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Could not recognize speech"

# Mock function for "live updates" (would connect to an API in production)
def get_live_updates():
    # Mock data for demonstration
    updates = [
        {"line": "Red", "status": "Normal", "message": "Operating on schedule"},
        {"line": "Blue", "status": "Delayed", "message": "Minor delays due to signal issues at Central"},
        {"line": "Green", "status": "Normal", "message": "Operating on schedule"},
        {"line": "Yellow", "status": "Alert", "message": "Track maintenance between Tech Hub and Market Street"},
    ]
    
    return updates

# Chatbot for metro queries
def metro_chatbot(query):
    query = query.lower()
    
    # Simple rule-based responses
    if "how" in query and "get" in query:
        for station in metro_data["stations"]:
            if station.lower() in query:
                return f"To get to {station}, you can take any of the following lines: {', '.join(metro_data['stations'][station]['lines'])}."
    
    elif "where" in query and "station" in query:
        for landmark in metro_data["landmarks"]:
            if landmark.lower() in query:
                station = metro_data["landmarks"][landmark]["nearest_station"]
                dist = metro_data["landmarks"][landmark]["distance"]
                return f"The nearest station to {landmark} is {station}, which is approximately {dist} km away."
    
    elif "line" in query:
        for line in ["Red", "Blue", "Green", "Yellow"]:
            if line.lower() in query:
                stations = [s for s, data in metro_data["stations"].items() if line in data["lines"]]
                return f"The {line} Line serves the following stations: {', '.join(stations)}."
    
    elif "fare" in query:
        return "Fares range from 10 to 35 currency units depending on distance traveled."
    
    elif "time" in query and "peak" in query:
        return "Peak hours are typically 7-9 AM and 5-7 PM on weekdays."
    
    else:
        return "I'm sorry, I couldn't understand your query. Try asking about specific stations, lines, or landmarks."

# Navigation interface - Main sidebar
def sidebar_interface():
    st.sidebar.title("🚇 Metro Navigation")
    
    # Dark mode toggle
    st.sidebar.checkbox("Dark Mode", key="dark_mode", on_change=set_theme)
    
    # Get current time for time-based features
    current_time = datetime.datetime.now()
    current_hour = current_time.hour
    
    # Navigation tabs
    nav_option = st.sidebar.selectbox(
        "Navigation Options",
        ["Route Planner", "Metro Map & Live Updates", "Station Finder", 
          "User Routes", "Metro Chatbot"]
    )
    
    return nav_option, current_time, current_hour

# Function to display route planner interface
def route_planner_interface(stations, current_time):
    st.header("Advanced Route Planner")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_station = st.selectbox("From:", list(stations.keys()), key="start_station")
        
    with col2:
        end_station = st.selectbox("To:", list(stations.keys()), key="end_station")
        
    with col3:
        planning_mode = st.radio("Route Type:", ["Fastest", "Cheapest", "Multiple Options", "Time-Based"])
    
    # Common parameters
    num_routes = 1
    if planning_mode == "Multiple Options":
        num_routes = st.slider("Number of route options:", 1, 5, 3)
    
    # Time settings for time-based routing
    if planning_mode == "Time-Based":
        departure_time = st.time_input("Departure Time:", current_time)
        departure_datetime = datetime.datetime.combine(datetime.date.today(), departure_time)
    else:
        departure_datetime = current_time
    
    if st.button("Find Routes"):
        if start_station == end_station:
            st.error("Start and end stations must be different")
        else:
            with st.spinner("Calculating best routes..."):
                if planning_mode == "Fastest":
                    routes = [find_shortest_path(metro_data["graph"], start_station, end_station, "time")]
                elif planning_mode == "Cheapest":
                    routes = [find_shortest_path(metro_data["graph"], start_station, end_station, "fare")]
                elif planning_mode == "Multiple Options":
                    routes = find_multiple_routes(metro_data["graph"], start_station, end_station, num_routes)
                else:  # Time-Based
                    routes = find_time_based_routes(metro_data["graph"], start_station, end_station, departure_datetime)
                
                # Display routes
                if routes and all(routes):
                    display_routes(routes, metro_data["stations"])
                else:
                    st.error("No route found between the selected stations")
    
    # Save route feature
    st.subheader("Save Your Frequent Routes")
    save_col1, save_col2 = st.columns([3, 1])
    
    with save_col1:
        route_name = st.text_input("Route Name:", placeholder="My Daily Commute")
    
    with save_col2:
        if st.button("Save Route") and route_name:
            new_route = {
                "name": route_name,
                "start": start_station,
                "end": end_station,
                "mode": planning_mode
            }
            st.session_state.user_routes.append(new_route)
            st.success(f"Route '{route_name}' saved successfully!")

# Function to display routes
def display_routes(routes, stations):
    for i, route in enumerate(routes):
        if route:
            st.subheader(f"Route Option {i+1}")
            
            # Create columns for details and map
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Route summary
                st.write(f"**Stations:** {' → '.join(route['path'])}")
                st.write(f"**Travel Time:** {route['total_time']} minutes")
                st.write(f"**Fare:** {route['total_fare']} units")
                
                # Show line transfers
                transfers = []
                current_lines = set()
                
                for i in range(len(route['path'])):
                    station = route['path'][i]
                    station_lines = set(stations[station]['lines'])
                    
                    if i == 0:  # First station
                        current_lines = station_lines
                    else:
                        # Check if we need to transfer
                        common_lines = current_lines.intersection(station_lines)
                        
                        if not common_lines:  # No common lines, need to transfer
                            transfer_station = route['path'][i-1]
                            transfers.append(f"Transfer at {transfer_station} from {', '.join(current_lines)} to {', '.join(station_lines)}")
                            current_lines = station_lines
                        else:
                            current_lines = common_lines
                
                if transfers:
                    st.write("**Transfers:**")
                    for transfer in transfers:
                        st.write(f"- {transfer}")
                else:
                    st.write("**No transfers required**")
            
            with col2:
                # Create map with highlighted route
                m = create_metro_map(highlighted_path=route['path'])
                folium_static(m)
        else:
            st.error("No valid route found")

# Function to display metro map and live updates
def metro_map_interface(current_hour):
    st.header("Metro Map & Live Updates")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_traffic = st.checkbox("Show Traffic Heatmap")
        selected_lines = st.multiselect("Filter by Lines:", ["Red", "Blue", "Green", "Yellow"], default=["Red", "Blue", "Green", "Yellow"])
        
        # Create filtered graph
        if len(selected_lines) < 4:  # If not all lines are selected
            filtered_graph = filter_by_lines(selected_lines)
            # We would need to adjust the map creation to use the filtered graph
            m = create_metro_map(traffic_heatmap=show_traffic, current_hour=current_hour)
        else:
            m = create_metro_map(traffic_heatmap=show_traffic, current_hour=current_hour)
        
        folium_static(m)
    
    with col2:
        st.subheader("Live Updates")
        updates = get_live_updates()
        
        for update in updates:
            if update["status"] == "Normal":
                status_color = "green"
            elif update["status"] == "Delayed":
                status_color = "orange"
            else:  # Alert
                status_color = "red"
                
            st.markdown(f"""
            <div style="padding: 10px; border-left: 5px solid {status_color}; margin-bottom: 10px;">
                <strong>{update['line']} Line:</strong> {update['status']}<br>
                {update['message']}
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

# Station finder interface
def station_finder_interface():
    st.header("Station Finder")
    
    finder_tab1, finder_tab2 = st.tabs(["Find by Location", "Find by Landmark"])
    
    with finder_tab1:
        st.subheader("Find Nearest Station")
        
        col1, col2 = st.columns(2)
        with col1:
            user_lat = st.number_input("Your Latitude:", value=28.6292, format="%.4f")
        with col2:
            user_lng = st.number_input("Your Longitude:", value=77.2074, format="%.4f")
        
        if st.button("Find Nearest Station", key="find_nearest"):
            with st.spinner("Finding nearest station..."):
                nearest, distance = find_nearest_station(user_lat, user_lng)
                
                st.success(f"The nearest station is **{nearest}**, which is approximately **{distance:.2f} km** from your location.")
                
                # Show on map
                m = folium.Map(location=[user_lat, user_lng], zoom_start=14)
                
                # Add user marker
                folium.Marker(
                    [user_lat, user_lng],
                    popup="Your Location",
                    icon=folium.Icon(icon="user", prefix="fa", color="red")
                ).add_to(m)
                
                # Add station marker
                station_pos = metro_data["stations"][nearest]["position"]
                folium.Marker(
                    station_pos,
                    popup=nearest,
                    icon=folium.Icon(icon="subway", prefix="fa", color="blue")
                ).add_to(m)
                
                # Add line between user and station
                folium.PolyLine(
                    [[user_lat, user_lng], station_pos],
                    color="purple",
                    weight=3,
                    opacity=0.7,
                    dash_array="10"
                ).add_to(m)
                
                folium_static(m)
    
    with finder_tab2:
        st.subheader("Find Station by Landmark")
        
        landmark_input = st.text_input("Enter landmark name:", placeholder="e.g., Museum, Park, Mall")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Search", key="landmark_search"):
                if landmark_input:
                    station = find_station_by_landmark(landmark_input)
                    if station:
                        landmark_matches = [lm for lm in metro_data["landmarks"].keys() 
                                          if landmark_input.lower() in lm.lower()]
                        
                        if landmark_matches:
                            landmark = landmark_matches[0]
                            distance = metro_data["landmarks"][landmark]["distance"]
                            
                            st.success(f"The landmark '{landmark}' is closest to **{station}** station, approximately **{distance:.1f} km** away.")
                            
                            # Show on map
                            station_pos = metro_data["stations"][station]["position"]
                            m = folium.Map(location=station_pos, zoom_start=14)
                            
                            # Add station marker
                            folium.Marker(
                                station_pos,
                                popup=station,
                                icon=folium.Icon(icon="subway", prefix="fa", color="blue")
                            ).add_to(m)
                            
                            # Add estimated landmark position (simple approximation)
                            # In reality, we would have actual landmark coordinates
                            import random
                            offset = distance * 0.009  # rough conversion to lat/lng degrees
                            landmark_pos = (
                                station_pos[0] + random.uniform(-offset, offset),
                                station_pos[1] + random.uniform(-offset, offset)
                            )
                            
                            folium.Marker(
                                landmark_pos,
                                popup=landmark,
                                icon=folium.Icon(icon="info-sign", color="green")
                            ).add_to(m)
                            
                            folium_static(m)
                        else:
                            st.error("No matching landmark found.")
                    else:
                        st.error("No station found near this landmark.")
                else:
                    st.warning("Please enter a landmark name.")
        
        with col2:
            st.subheader("Available Landmarks")
            for landmark, data in metro_data["landmarks"].items():
                st.write(f"- {landmark} (near {data['nearest_station']})")
    
    # Voice input feature
    st.subheader("Voice Input")
    if st.button("Use Voice Input"):
        with st.spinner("Listening..."):
            # In a real app, we would use the voice_to_text function
            # For demo purposes, just simulate the result
            time.sleep(2)
            detected_text = "University"
            st.info(f"Detected station: {detected_text}")
            
            # Find stations that match the voice input
            matches = [s for s in metro_data["stations"].keys() 
                     if detected_text.lower() in s.lower()]
            
            if matches:
                st.success(f"Found matching station: {matches[0]}")
            else:
                st.error("No matching station found. Please try again.")

# Traffic analysis interface
def traffic_analysis_interface():
    st.header("Traffic & Network Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Traffic Heatmap", "Travel Trends", "Network Centrality"])
    
    with analysis_tab1:
        st.subheader("Metro Traffic Heatmap")
        
        # Select hour for visualization
        selected_hour = st.slider("Hour of day:", 0, 23, 8)
        
        # Create heatmap
        m = create_metro_map(traffic_heatmap=True, current_hour=selected_hour)
        folium_static(m)
        
        st.info(f"Showing estimated passenger traffic at {selected_hour}:00")
        
        # Display traffic table
        traffic_data = []
        for station, hourly_traffic in metro_data["traffic_data"].items():
            traffic_data.append({
                "Station": station,
                "Traffic": hourly_traffic[selected_hour],
                "Lines": ", ".join(metro_data["stations"][station]["lines"])
            })
        
        traffic_df = pd.DataFrame(traffic_data).sort_values("Traffic", ascending=False)
        st.dataframe(traffic_df)
    
    with analysis_tab2:
        st.subheader("Travel Trends Analysis")
        
        # Get trend data
        avg_by_hour, busiest_stations = analyze_trends()
        
        # Plot average traffic by hour
        st.write("**Average Traffic by Hour**")
        fig = px.line(avg_by_hour, x="Hour", y="Traffic", 
                     title="Average Passenger Traffic Throughout the Day",
                     labels={"Traffic": "Average Passengers/Hour"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot busiest stations
        st.write("**Busiest Stations**")
        fig2 = px.bar(busiest_stations.head(10), x="Station", y="Traffic",
                     title="Top 10 Busiest Stations (Average Traffic)",
                     labels={"Traffic": "Average Passengers/Hour"})
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add some insights
        st.subheader("Key Insights")
        st.markdown("""
        - The network experiences two main peak periods:
          - Morning rush (7-9 AM)
          - Evening rush (5-7 PM)
        - Central and Market Street are consistently the busiest stations
        - Weekend traffic patterns differ significantly from weekdays
        - The Red Line carries the most passengers overall
        """)
    
    with analysis_tab3:
        st.subheader("Network Centrality Analysis")
        
        # Get centrality measures
        centrality_df = analyze_centrality()
        
        # Display centrality metrics
        st.write("**Station Centrality Measures**")
        st.dataframe(centrality_df.sort_values("Betweenness Centrality", ascending=False))
        
        # Plot network graph with centrality visualization
        st.write("**Network Graph Visualization**")
        G = metro_data["graph"]
        pos = nx.spring_layout(G, seed=42)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use betweenness centrality for node size
        node_sizes = [v * 3000 for v in centrality_df.set_index("Station")["Betweenness Centrality"]]
        
        # Use degree centrality for node color
        node_colors = list(centrality_df.set_index("Station")["Degree Centrality"])
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)
        
        plt.axis('off')
        st.pyplot(fig)
        
        # Add explanations
        st.markdown("""
        **Understanding Centrality Measures:**
        
        - **Degree Centrality**: Number of direct connections a station has.
        - **Betweenness Centrality**: How often a station acts as a bridge along the shortest path between other stations.
        - **Closeness Centrality**: How close a station is to all other stations in the network.
        
        Stations with high betweenness centrality are critical for network flow and are 
        more likely to cause widespread disruption if they experience issues.
        """)

# User routes interface
def user_routes_interface():
    st.header("Your Saved Routes")
    
    if not st.session_state.user_routes:
        st.info("You haven't saved any routes yet. Use the Route Planner to save your frequent routes.")
    else:
        for i, route in enumerate(st.session_state.user_routes):
            with st.expander(f"{route['name']}: {route['start']} to {route['end']}"):
                st.write(f"**From:** {route['start']}")
                st.write(f"**To:** {route['end']}")
                st.write(f"**Planning Mode:** {route['mode']}")
                
                if st.button("Load Route", key=f"load_route_{i}"):
                    # Set session state values for route planner
                    st.session_state.start_station = route['start']
                    st.session_state.end_station = route['end']
                    st.success("Route loaded! Go to Route Planner to view.")
                
                if st.button("Delete Route", key=f"delete_route_{i}"):
                    st.session_state.user_routes.pop(i)
                    st.rerun()
    
    # Add user-defined route feature
    st.subheader("Add Custom Route")
    st.write("Create a custom route by selecting stations in order:")
    
    # Allow user to select stations in sequence
    stations_list = list(metro_data["stations"].keys())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_station = st.selectbox("Select Station:", stations_list)
    
    with col2:
        if st.button("Add Station"):
            if "custom_route" not in st.session_state:
                st.session_state.custom_route = []
            
            st.session_state.custom_route.append(selected_station)
            st.rerun()
    
    with col3:
        if st.button("Clear") and "custom_route" in st.session_state:
            st.session_state.custom_route = []
            st.rerun()
    
    # Display current custom route
    if "custom_route" in st.session_state and st.session_state.custom_route:
        st.write("**Current route sequence:**", " → ".join(st.session_state.custom_route))
        
        if len(st.session_state.custom_route) >= 2:
            route_name = st.text_input("Route Name:", placeholder="My Custom Route")
            
            if st.button("Save Custom Route") and route_name:
                new_route = {
                    "name": route_name,
                    "start": st.session_state.custom_route[0],
                    "end": st.session_state.custom_route[-1],
                    "mode": "Custom",
                    "custom_path": st.session_state.custom_route
                }
                
                st.session_state.user_routes.append(new_route)
                st.success(f"Custom route '{route_name}' saved successfully!")
                
                # Clear current custom route
                st.session_state.custom_route = []
                st.rerun()

# Metro chatbot interface
def chatbot_interface():
    st.header("Metro Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Metro Assistant:** {message['content']}")
    
    # Chat input
    user_query = st.text_input("Ask me anything about the metro system:", placeholder="e.g., How do I get to City Park?")
    
    if st.button("Send") and user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Get response from chatbot
        response = metro_chatbot(user_query)
        
        # Add response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Clear input and refresh
        st.rerun()
    
    # AI suggestions based on time of day
    current_hour = datetime.datetime.now().hour
    suggestions = get_predictive_suggestions(current_hour)
    
    st.subheader("Smart Suggestions")
    st.info(suggestions["message"])
    
    if suggestions["from_stations"] and suggestions["to_stations"]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Popular origins now:**")
            for station in suggestions["from_stations"]:
                st.write(f"- {station}")
        
        with col2:
            st.write("**Popular destinations now:**")
            for station in suggestions["to_stations"]:
                st.write(f"- {station}")
    
    # Sample questions
    st.subheader("Popular Questions")
    sample_questions = [
        "How do I get to Central station?",
        "Which line goes to University?",
        "Where is the National Museum?",
        "What are the peak hours?",
        "How much is the fare from Central to Riverside?"
    ]
    
    for question in sample_questions:
        if st.button(question):
            # Add question to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # Get response
            response = metro_chatbot(question)
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Refresh
            st.rerun()

# Main application function
def main():
    # Apply theme based on dark mode setting
    set_theme()
    
    # Display header
    st.title("🚇 Metro Navigation System")
    
    # Sidebar navigation
    nav_option, current_time, current_hour = sidebar_interface()
    
    # Main content based on selected navigation
    if nav_option == "Route Planner":
        route_planner_interface(metro_data["stations"], current_time)
    
    elif nav_option == "Metro Map & Live Updates":
        metro_map_interface(current_hour)
    
    elif nav_option == "Station Finder":
        station_finder_interface()
    
    elif nav_option == "Traffic Analysis":
        traffic_analysis_interface()
    
    elif nav_option == "User Routes":
        user_routes_interface()
    
    elif nav_option == "Metro Chatbot":
        chatbot_interface()

if __name__ == "__main__":
    main()

