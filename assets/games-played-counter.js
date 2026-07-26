(function () {
  var DURATION_MS = 1200;

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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }

  // Dash re-renders the landing page on language change / client routing.
  var observer = new MutationObserver(function () {
    scan();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
