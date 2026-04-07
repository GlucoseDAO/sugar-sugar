/**
 * Auto-resize srcdoc iframes marked with data-autosize="true".
 * Same-origin srcdoc allows direct access to contentDocument.
 */
(function () {
    function resizeAll() {
        var frames = document.querySelectorAll('iframe[data-autosize="true"]');
        for (var i = 0; i < frames.length; i++) {
            var f = frames[i];
            try {
                var doc = f.contentDocument || (f.contentWindow && f.contentWindow.document);
                if (doc && doc.body) {
                    var h = doc.documentElement.scrollHeight;
                    if (h > 0) f.style.height = h + "px";
                    
                    // Sync dark mode from parent
                    if (document.body.classList.contains('dark-mode')) {
                        doc.body.classList.add('dark-mode');
                    } else {
                        doc.body.classList.remove('dark-mode');
                    }
                }
            } catch (_) { /* cross-origin — skip */ }
        }
    }

    function setup() {
        var frames = document.querySelectorAll('iframe[data-autosize="true"]');
        for (var i = 0; i < frames.length; i++) {
            frames[i].addEventListener("load", resizeAll);
        }
        resizeAll();
    }

    // Dash may render iframes after DOMContentLoaded, so observe mutations.
    var observer = new MutationObserver(function () {
        setup();
    });
    observer.observe(document.body || document.documentElement, {childList: true, subtree: true});

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setup);
    } else {
        setup();
    }
    // Fallback poll for late-loading content inside the iframe.
    setInterval(resizeAll, 500);
})();
