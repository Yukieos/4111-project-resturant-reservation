# 🍽️ Restaurant Reservation System

This project was built as part of COMS W4111 - Introduction to Databases.

The system allows users to browse restaurants, view available time slots, and make reservations through a web interface.

### 🔧 Tech Stack
- Flask (Python)
- PostgreSQL (schema & SQL queries)
- HTML/CSS/JS (with Jinja templates)
- Google Maps API + Yelp integration
- GCP for database deployment

### 💡 Features
- Restaurant search and filtering
- Live reservation booking
- Admin page for restaurant owners
- **Restaurant Recommendations** *(Self-implemented additional feature outside course requirements)*

#### **Recommendation System: **

The system will provide personalized restaurant suggestions based on user dining history using:
- **Markov Chain Analysis**: Predicts next cuisine preference based on user's dining patterns
- **Multi-Factor Scoring Algorithm**: Combines multiple factors for optimal recommendations
- **Content-based filtering**: Uses similarity metrics for personalized suggestions
  - Cuisine Match (40%)
  - Price Similarity (30%): via cosine distance
  - Location Preference (20%) using simulated coordinates
  - Rating Weight (10%)

### **How to Use:**
1. Navigate to the /recommendations endpoint
2. Enter your user credentials (last name + phone number)
3. Get personalized restaurant recommendations with match scores
4. Click through to make reservations at recommended restaurants

---

> Final project submitted to Columbia University, Spring 2025
