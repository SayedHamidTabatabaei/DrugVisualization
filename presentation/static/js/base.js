document.addEventListener('DOMContentLoaded', function() {
    const menuItems = document.querySelectorAll('.menu-item');

    menuItems.forEach(function(menuItem) {
        menuItem.addEventListener('click', function(event) {
            const submenu = menuItem.nextElementSibling;

            if (submenu && submenu.classList.contains('submenu')) {
                event.preventDefault(); // Prevent default link behavior
                submenu.style.display = (submenu.style.display === 'none' || submenu.style.display === '') ? 'block' : 'none';
            }
        });
    });
});

function remove_incorrect_jobs(){

    showSpinner();

    fetch(`/job/incorrect_job_delete`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            hideSpinner(true);

        } else {
            console.log('Error: No data found.');
            hideSpinner(false);
        }
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
}

function start_job(){

    showSpinner();

    fetch(`/job/start_job`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            hideSpinner(true);

        } else {
            console.log('Error: No data found.');
            hideSpinner(false);
        }
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
}
