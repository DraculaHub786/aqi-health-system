// State Management
let currentProfile = 'general';
let currentLocation = '';

// API Configuration - Now points to same server
const API_URL = '/api';

// DOM Elements
const profileCards = document.querySelectorAll('.profile-card');
const locationInput = document.getElementById('locationInput');
const checkAQIBtn = document.getElementById('checkAQIBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultsContainer = document.getElementById('resultsContainer');
const errorMessage = document.getElementById('errorMessage');

// Initialize Event Listeners
function initializeApp() {
    profileCards.forEach(card => {
        card.addEventListener('click', handleProfileSelect);
    });
    
    checkAQIBtn.addEventListener('click', handleCheckAQI);
    
    locationInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleCheckAQI();
        }
    });
    
    console.log('🌍 AI AQI Health System Initialized');
}

// Handle Profile Selection
function handleProfileSelect(e) {
    profileCards.forEach(card => card.classList.remove('active'));
    e.currentTarget.classList.add('active');
    currentProfile = e.currentTarget.dataset.profile;
    console.log('Profile selected:', currentProfile);
}

// Handle AQI Check
async function handleCheckAQI() {
    const location = locationInput.value.trim();
    
    if (!location) {
        showError('Please enter a location');
        return;
    }
    
    currentLocation = location;
    setLoading(true);
    hideError();
    
    try {
        console.log('Fetching AQI data for:', location);
        
        const response = await fetch(`${API_URL}/aqi`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                location: location,
                userProfile: currentProfile
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to fetch AQI data');
        }
        
        const data = await response.json();
        console.log('AQI data received:', data);
        displayResults(data);
        
    } catch (error) {
        showError('Unable to fetch AQI data. Please try again.');
        console.error('Error:', error);
    } finally {
        setLoading(false);
    }
}

// Display Results
function displayResults(data) {
    resultsContainer.classList.remove('hidden');
    
    // Update AQI Card
    document.getElementById('aqiValue').textContent = data.aqi;
    document.getElementById('aqiValue').style.color = data.color;
    document.getElementById('aqiCategory').textContent = data.category;
    document.getElementById('aqiCategory').style.color = data.color;
    document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
    
    // Update Health Score
    const score = calculateHealthScore(data.aqi);
    document.getElementById('scoreBar').style.width = `${score}%`;
    document.getElementById('scoreValue').textContent = `${score}/100`;
    
    // Update Pollutants
    displayPollutants(data.pollutants);
    
    // Update NLP Explanation
    document.getElementById('nlpExplanation').textContent = data.explanation;
    
    // Update Recommendations
    displayRecommendations(data.recommendations);
    
    // Update Time Slots
    displayTimeSlots(data.timeSlots);
    
    // Update Health Tips
    displayHealthTips(data.healthTips);
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Pollutants
function displayPollutants(pollutants) {
    const grid = document.getElementById('pollutantGrid');
    grid.innerHTML = '';
    
    for (const [name, value] of Object.entries(pollutants)) {
        const item = document.createElement('div');
        item.className = 'pollutant-item';
        item.innerHTML = `
            <div class="pollutant-name">${name.toUpperCase()}</div>
            <div class="pollutant-value">${value}</div>
        `;
        grid.appendChild(item);
    }
}

// Display Recommendations
function displayRecommendations(recommendations) {
    const grid = document.getElementById('recommendationsGrid');
    grid.innerHTML = '';
    
    recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `
            <div class="rec-icon">${rec.icon}</div>
            <div class="rec-name">${rec.name}</div>
            <div class="rec-duration">⏱ ${rec.duration}</div>
            <span class="rec-intensity">${rec.intensity}</span>
        `;
        grid.appendChild(item);
    });
}

// Display Time Slots
function displayTimeSlots(timeSlots) {
    const container = document.getElementById('timeSlotsContainer');
    container.innerHTML = '';
    
    timeSlots.forEach(slot => {
        const item = document.createElement('div');
        item.className = 'timeslot-item';
        item.style.backgroundColor = slot.color + '20';
        item.style.borderLeft = `5px solid ${slot.color}`;
        item.innerHTML = `
            <div class="timeslot-time">${slot.time}</div>
            <div class="timeslot-aqi" style="color: ${slot.color}">${slot.aqi}</div>
            <div class="timeslot-category">${slot.category}</div>
        `;
        container.appendChild(item);
    });
}

// Display Health Tips
function displayHealthTips(tips) {
    const list = document.getElementById('healthTips');
    list.innerHTML = '';
    
    tips.forEach(tip => {
        const item = document.createElement('li');
        item.textContent = tip;
        list.appendChild(item);
    });
}

// Helper Functions
function calculateHealthScore(aqi) {
    if (aqi <= 50) return 100;
    if (aqi <= 100) return 80;
    if (aqi <= 150) return 60;
    if (aqi <= 200) return 40;
    if (aqi <= 300) return 20;
    return 5;
}

function setLoading(loading) {
    checkAQIBtn.disabled = loading;
    if (loading) {
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => hideError(), 5000);
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// Initialize App
document.addEventListener('DOMContentLoaded', initializeApp);