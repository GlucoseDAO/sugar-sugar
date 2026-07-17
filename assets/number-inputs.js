// Prevent wheel events from changing number input values
// Only allow keyboard input or clicking the spinner arrows

document.addEventListener('DOMContentLoaded', function() {
  // Target specific number inputs by ID
  const numberInputIds = ['age-input', 'cgm-duration-input', 'diabetes-duration-input'];
  
  numberInputIds.forEach(inputId => {
    const input = document.getElementById(inputId);
    if (input) {
      input.addEventListener('wheel', function(e) {
        e.preventDefault();
      }, { passive: false });
    }
  });
});

// Also handle dynamically created or re-rendered inputs (for Dash updates)
const observer = new MutationObserver(function() {
  const numberInputIds = ['age-input', 'cgm-duration-input', 'diabetes-duration-input'];
  
  numberInputIds.forEach(inputId => {
    const input = document.getElementById(inputId);
    if (input && !input.hasAttribute('data-wheel-disabled')) {
      input.setAttribute('data-wheel-disabled', 'true');
      input.addEventListener('wheel', function(e) {
        e.preventDefault();
      }, { passive: false });
    }
  });
});

// Start observing the document for changes
observer.observe(document.body, {
  childList: true,
  subtree: true
});
