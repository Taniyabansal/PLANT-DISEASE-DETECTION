const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {
        const imageURL = URL.createObjectURL(file);

        preview.src = imageURL;
        preview.style.display = "block";
    }
});