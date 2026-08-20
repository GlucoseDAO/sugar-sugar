/**
 * Auto-resize srcdoc iframes marked with data-autosize="true".
 * Same-origin srcdoc allows direct access to contentDocument.
 *
 * Only `/about` renders such an iframe (static_markdown_autosize_iframe), but
 * this file is served on every page, so it must cost nothing elsewhere. The
 * previous version ran `setInterval(resizeAll, 500)` forever on every page and
 * re-added a `load` listener to every frame on every DOM mutation (an unbounded
 * listener leak on a Dash app that re-renders constantly). Both are gone: sizing
 * is now event-driven (load + ResizeObserver on the inner document + window
 * resize), with a short bounded poll to catch late webfont/image reflow.
 */
(function () {
    var WIRED_ATTR = "data-autosize-wired";
    var SELECTOR = 'iframe[data-autosize="true"]';
    var SETTLE_POLL_MS = 250;
    var SETTLE_POLL_TICKS = 8;   // ~2s, then it stops for good
    var scanQueued = false;

    function innerDoc(frame) {
        try {
            return frame.contentDocument || (frame.contentWindow && frame.contentWindow.document);
        } catch (_) {
            return null;   // cross-origin — skip
        }
    }

    function resizeFrame(frame) {
        var doc = innerDoc(frame);
        if (!doc || !doc.body) {
            return;
        }
        var height = doc.documentElement.scrollHeight;
        if (height > 0 && frame.style.height !== height + "px") {
            frame.style.height = height + "px";
        }
    }

    function resizeAll() {
        var frames = document.querySelectorAll(SELECTOR);
        for (var i = 0; i < frames.length; i++) {
            resizeFrame(frames[i]);
        }
    }

    /** Bounded catch-up for content that lays out after `load` (fonts, images). */
    function settle(frame) {
        var ticks = 0;
        var timer = setInterval(function () {
            resizeFrame(frame);
            if (++ticks >= SETTLE_POLL_TICKS) {
                clearInterval(timer);
            }
        }, SETTLE_POLL_MS);
    }

    function wire(frame) {
        if (frame.getAttribute(WIRED_ATTR) === "1") {
            return;
        }
        frame.setAttribute(WIRED_ATTR, "1");
        frame.addEventListener("load", function () {
            // A srcdoc change (language switch on /about) replaces the inner
            // document, so the observer from the previous one is now watching a
            // detached node. Clear the flag or observeContent() would no-op and
            // late reflow past the settle window would never re-fit.
            frame._autosizeObserved = false;
            resizeFrame(frame);
            observeContent(frame);
            settle(frame);
        });
        // srcdoc frames are often already parsed by the time we get here.
        resizeFrame(frame);
        observeContent(frame);
        settle(frame);
    }

    /** Re-fit when the iframe's own content reflows — replaces the forever poll. */
    function observeContent(frame) {
        if (typeof ResizeObserver === "undefined" || frame._autosizeObserved) {
            return;
        }
        var doc = innerDoc(frame);
        if (!doc || !doc.documentElement) {
            return;
        }
        frame._autosizeObserved = true;
        var observer = new ResizeObserver(function () {
            resizeFrame(frame);
        });
        observer.observe(doc.documentElement);
    }

    function scan() {
        var frames = document.querySelectorAll(SELECTOR);
        for (var i = 0; i < frames.length; i++) {
            wire(frames[i]);
        }
    }

    /** One scan per animation frame at most, however many mutations land. */
    function queueScan() {
        if (scanQueued) {
            return;
        }
        scanQueued = true;
        window.requestAnimationFrame(function () {
            scanQueued = false;
            scan();
        });
    }

    var observer = new MutationObserver(queueScan);
    observer.observe(document.body || document.documentElement, {childList: true, subtree: true});

    window.addEventListener("resize", resizeAll);

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", scan);
    } else {
        scan();
    }
})();
