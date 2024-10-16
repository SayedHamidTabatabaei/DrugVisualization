function fill_loss_dropdown() {
    const lossSelect = document.getElementById('lossFunctionSelect');
    const lossContainer = document.getElementById('loss-container');
    const selectedModel = document.getElementById("trainModelSelect").value;

    lossSelect.innerHTML = '';
    fetch(`/training/fillLossFunctions?trainModel=${selectedModel}`)
        .then(response => response.json())
        .then(data => {

            if(data && data.length > 0){
                lossContainer.style.display='flex';
            } else{
                lossContainer.style.display='none';
            }

            data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.value;
                option.text = model.name;
                lossSelect.appendChild(option);
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
}