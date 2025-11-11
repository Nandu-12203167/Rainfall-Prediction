document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resetBtn = document.getElementById('resetBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultSection = document.getElementById('resultSection');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        predictBtn.disabled = true;

        const formData = {
            pressure: parseFloat(document.getElementById('pressure').value),
            maxtemp: parseFloat(document.getElementById('maxtemp').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            mintemp: parseFloat(document.getElementById('mintemp').value),
            dewpoint: parseFloat(document.getElementById('dewpoint').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            cloud: parseFloat(document.getElementById('cloud').value),
            sunshine: parseFloat(document.getElementById('sunshine').value),
            winddirection: parseFloat(document.getElementById('winddirection').value),
            windspeed: parseFloat(document.getElementById('windspeed').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error occurred'));
            }
        } catch (error) {
            alert('Error connecting to the server: ' + error.message);
        } finally {
            predictBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', function() {
        form.reset();
        resultSection.classList.remove('show');
    });

    function displayResult(data) {
        const predictionValue = document.getElementById('predictionValue');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceFill = document.getElementById('confidenceFill');
        const noRainProb = document.getElementById('noRainProb');
        const rainProb = document.getElementById('rainProb');
        const resultIcon = document.getElementById('resultIcon');

        predictionValue.textContent = data.prediction;
        confidenceValue.textContent = data.confidence.toFixed(1) + '%';
        confidenceFill.style.width = data.confidence + '%';
        noRainProb.textContent = data.probability.no_rain.toFixed(1) + '%';
        rainProb.textContent = data.probability.rain.toFixed(1) + '%';

        resultIcon.className = 'result-icon';
        if (data.prediction === 'Rain') {
            resultIcon.classList.add('rain');
            resultIcon.textContent = '🌧️';
        } else {
            resultIcon.classList.add('no-rain');
            resultIcon.textContent = '☀️';
        }

        resultSection.classList.add('show');

        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
