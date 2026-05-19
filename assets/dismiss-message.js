document.addEventListener('click', function(e) {
    if (e.target.closest('.message > .close.icon')) {
        var msg = e.target.closest('.message');
        if (msg) msg.style.display = 'none';
    }
});
