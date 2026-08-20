// Count-up animation for the landing page's "games played so far" figure.
//
// The counter exists only on the landing page, but this file is served
// everywhere. The MutationObserver below therefore does the least it can: at
// most one querySelectorAll per animation frame, however many mutations Dash
// fires, and it disconnects once a counter has been animated (the element is
// re-created on language change, so it re-arms via the Dash navigation render).
(function () {
  var DURATION_MS = 1200;
  var scanQueued = false;

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function animateCount(el) {
    if (!el || el.dataset.animated === "1") {
      return;
    }
    var target = parseInt(el.getAttribute("data-target") || "0", 10);
    if (!isFinite(target) || target < 0) {
      target = 0;
    }
    el.dataset.animated = "1";
    if (target === 0) {
      el.textContent = "0";
      return;
    }

    var start = null;
    function frame(ts) {
      if (start === null) {
        start = ts;
      }
      var progress = Math.min(1, (ts - start) / DURATION_MS);
      var value = Math.round(easeOutCubic(progress) * target);
      el.textContent = String(value);
      if (progress < 1) {
        window.requestAnimationFrame(frame);
      } else {
        el.textContent = String(target);
      }
    }
    window.requestAnimationFrame(frame);
  }

  function scan() {
    var nodes = document.querySelectorAll(".games-played-count");
    for (var i = 0; i < nodes.length; i++) {
      animateCount(nodes[i]);
    }
  }

  /** Collapse a burst of Dash mutations into a single scan per frame. */
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }

  // Dash re-renders the landing page on language change / client routing.
  var observer = new MutationObserver(queueScan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
