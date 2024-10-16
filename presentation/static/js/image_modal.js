
function openImageModal(img) {
    let modal = document.getElementById("imageModal");
    let modalImg = document.getElementById("enlargedImage");

    modal.style.display = "block";
    modalImg.src = img.src;
}
function openImageModalBySrc(src) {
    let modal = document.getElementById("imageModal");
    let modalImg = document.getElementById("enlargedImage");

    modal.style.display = "block";
    modalImg.src = src;
}

function closeImageModal() {
    let modal = document.getElementById("imageModal");
    modal.style.display = "none";
}