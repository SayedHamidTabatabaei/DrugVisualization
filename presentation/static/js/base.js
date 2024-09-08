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
