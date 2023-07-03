
function show_image() {
    var originalImg = document.getElementById("modal-original-img");
    //show on show.bs.modal
    $('#show-image-modal').on('shown.bs.modal', function(e) {
        var link = $(e.relatedTarget)
        modal = $(this);
        original_image = link.data('original-image');
        const originalImgURL = "/static/footage/"+original_image+"";
        originalImg.src = originalImgURL;
    });
}