function toggleDropdowns(checkbox, containerId) {
    let container = document.getElementById(containerId);
    if (checkbox.checked) {
        container.style.display = 'flex';
    } else {
        container.style.display = 'none';
    }
}