/* Grow the cartesian (drawn) area on prediction + results charts.
   CSS can hide titles but cannot enlarge Plotly's plot box — relayout
   + resize does. Margin numbers must match glucose.py _COMPACT_MARGIN
   / _DESKTOP_MARGIN. Runs in every environment (debug / staging /
   production / Chrome): this file is a static asset, not gated on a
   build flag. */
(function () {
  var COMPACT = { t: 2, b: 40, l: 36, r: 4, pad: 0 };
  var DESKTOP = { t: 8, b: 36, r: 8, pad: 0 };

  function isGlucosePlot(gd) {
    if (!gd) {
      return false;
    }
    var id = gd.id || "";
    if (id === "glucose-graph-graph" || id === "ending-static-graph") {
      return true;
    }
    return !!(gd.closest && gd.closest(".glucose-chart-shell"));
  }

  function isMobileChart() {
    // Same predicate as the `mobile-device` <html> class in app.py.
    // A desktop touchscreen is pointer:coarse but hover:hover / wide —
    // treating that as a phone rotated the HH:MM ticks into a 20px strip
    // and clipped them on the desktop /prediction card.
    var root = document.documentElement;
    if (root.classList.contains("mobile-device")) {
      return true;
    }
    if (!window.matchMedia) {
      return false;
    }
    try {
      return window.matchMedia("(pointer: coarse)").matches
        && window.matchMedia("(max-device-width: 1024px)").matches;
    } catch (err) {
      return false;
    }
  }

  function growPlot(gd) {
    if (!gd || !window.Plotly || !isGlucosePlot(gd)) {
      return;
    }
    var mobile = isMobileChart();
    var pack = mobile ? COMPACT : DESKTOP;
    if (!gd._sugarPlotGrown) {
      gd._sugarPlotGrown = true;
      var payload = {
        "xaxis.automargin": false,
        "yaxis.automargin": false,
        "margin.t": pack.t,
        "margin.b": pack.b,
        "margin.r": pack.r,
        "margin.pad": pack.pad,
        "legend.y": 0.99,
        "legend.yanchor": "top",
        "xaxis.title.text": "",
        "xaxis.title.standoff": 0
      };
      if (mobile) {
        payload["margin.l"] = COMPACT.l;
        payload["yaxis.title.text"] = "";
        payload["xaxis.tickangle"] = -90;
        payload["xaxis.tickfont.size"] = 8;
        payload["yaxis.tickfont.size"] = 9;
        payload["xaxis.ticklen"] = 2;
      } else {
        payload["xaxis.tickangle"] = 0;
        payload["xaxis.tickfont.size"] = 11;
      }
      window.Plotly.relayout(gd, payload);
    }
    window.Plotly.Plots.resize(gd);
  }

  function growAll() {
    document.querySelectorAll(".js-plotly-plot").forEach(growPlot);
  }

  document.addEventListener("plotly_afterplot", function (event) {
    growPlot(event.target);
  }, true);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", growAll);
  } else {
    growAll();
  }
  window.addEventListener("resize", growAll);
})();
