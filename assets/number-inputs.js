// Prevent the mouse wheel from silently changing number input values
// (age, CGM duration, diabetes duration). Keyboard and the spinner arrows
// still work.
//
// A browser only applies wheel-to-value while the number input has focus, so
// the guard is attached on focus and removed on blur. The previous version kept
// a MutationObserver on document.body (childList + subtree) alive for the whole
// session on every page just to re-attach these three listeners after a Dash
// re-render -- pure overhead on a page that re-renders on every callback.

(function () {
  'use strict';

  var GUARDED_IDS = ['age-input', 'cgm-duration-input', 'diabetes-duration-input'];

  function blockWheel(event) {
    event.preventDefault();
  }

  function isGuarded(node) {
    return !!node && GUARDED_IDS.indexOf(node.id) !== -1;
  }

  document.addEventListener('focusin', function (event) {
    if (isGuarded(event.target)) {
      event.target.addEventListener('wheel', blockWheel, { passive: false });
    }
  }, true);

  document.addEventListener('focusout', function (event) {
    if (isGuarded(event.target)) {
      event.target.removeEventListener('wheel', blockWheel, { passive: false });
    }
  }, true);
})();
