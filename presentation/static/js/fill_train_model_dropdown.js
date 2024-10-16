function fill_train_model_dropdown(default_text="--Select a Model--") {

    const selectedScenario = document.getElementById("scenarioSelect").value;
    const similaritySelect = document.getElementById('trainModelSelect');

    similaritySelect.innerHTML = `<option value="">${default_text}</option>`;
    fetch(`/training/fillTrainingModels?scenario=${selectedScenario}`)
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create option elements
            data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.text = model.name;
                similaritySelect.appendChild(option);
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
}