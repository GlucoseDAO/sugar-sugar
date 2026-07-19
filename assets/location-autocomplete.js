// Location autocomplete for #location-input on /startup.
// Loaded from assets/; init is also triggered by a clientside callback on navigation.

(function () {
  'use strict';

  var INPUT_ID = 'location-input';
  var LOCALE_STORE_ID = 'interface-language';
  var DATA_URL = '/assets/location-suggestions.json';
  var MIN_CHARS = 2;
  var MAX_RESULTS = 8;
  var DEBOUNCE_MS = 120;
  var SUPPORTED_LOCALES = ['de', 'en', 'es', 'fr', 'ro', 'ru', 'uk', 'zh'];

  var places = null;
  var loadPromise = null;
  var dropdown = null;
  var debounceTimer = null;
  var activeIndex = -1;
  var suppressNextInput = false;

  function normalizeLocale(raw) {
    if (typeof raw !== 'string') {
      return 'en';
    }
    return SUPPORTED_LOCALES.indexOf(raw) !== -1 ? raw : 'en';
  }

  function getLocale() {
    try {
      var raw = window.localStorage.getItem(LOCALE_STORE_ID);
      if (raw) {
        var parsed = JSON.parse(raw);
        if (typeof parsed === 'string') {
          return normalizeLocale(parsed);
        }
      }
    } catch (e) {
      /* ignore */
    }
    return 'en';
  }

  function asciiFold(text) {
    if (!text) {
      return '';
    }
    return text
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase();
  }

  function loadPlaces() {
    if (places) {
      return Promise.resolve(places);
    }
    if (loadPromise) {
      return loadPromise;
    }
    loadPromise = fetch(DATA_URL)
      .then(function (response) {
        if (!response.ok) {
          throw new Error('Failed to load location suggestions');
        }
        return response.json();
      })
      .then(function (data) {
        if (!Array.isArray(data)) {
          places = [];
          return places;
        }
        // Accept both legacy string rows and the current object rows.
        places = data.map(function (row) {
          if (typeof row === 'string') {
            return {
              canonical: row,
              labels: { en: row },
              search: [row.toLowerCase()],
            };
          }
          return row;
        });
        return places;
      })
      .catch(function () {
        places = [];
        return places;
      });
    return loadPromise;
  }

  function displayLabel(place) {
    var locale = getLocale();
    if (place.labels && place.labels[locale]) {
      return place.labels[locale];
    }
    if (place.labels && place.labels.en) {
      return place.labels.en;
    }
    return place.canonical || '';
  }

  function placeMatches(place, q, qFold) {
    var tokens = place.search || [];
    for (var i = 0; i < tokens.length; i++) {
      var token = String(tokens[i]).toLowerCase();
      if (token.indexOf(q) === 0 || (qFold && token.indexOf(qFold) === 0)) {
        return 2;
      }
    }
    for (var j = 0; j < tokens.length; j++) {
      var tokenContains = String(tokens[j]).toLowerCase();
      if (tokenContains.indexOf(q) !== -1 || (qFold && tokenContains.indexOf(qFold) !== -1)) {
        return 1;
      }
    }
    return 0;
  }

  function placeSortKey(a, b) {
    var rankA = typeof a.rank === 'number' ? a.rank : 1000;
    var rankB = typeof b.rank === 'number' ? b.rank : 1000;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return displayLabel(a).localeCompare(displayLabel(b));
  }

  function filterPlaces(query) {
    if (!places || !query) {
      return [];
    }
    var q = query.trim().toLowerCase();
    if (q.length < MIN_CHARS) {
      return [];
    }
    var qFold = asciiFold(q);

    var prefix = [];
    var contains = [];
    var seen = {};

    for (var i = 0; i < places.length; i++) {
      var place = places[i];
      var rank = placeMatches(place, q, qFold);
      if (!rank) {
        continue;
      }
      var label = displayLabel(place);
      if (!label || seen[label]) {
        continue;
      }
      seen[label] = true;
      if (rank === 2) {
        prefix.push(place);
      } else {
        contains.push(place);
      }
    }

    prefix.sort(placeSortKey);
    contains.sort(placeSortKey);

    return prefix.concat(contains).slice(0, MAX_RESULTS);
  }

  function hideDropdown() {
    activeIndex = -1;
    if (dropdown) {
      dropdown.remove();
      dropdown = null;
    }
  }

  function getInputElement() {
    var node = document.getElementById(INPUT_ID);
    if (!node) {
      return null;
    }
    if (node.tagName === 'INPUT') {
      return node;
    }
    return node.querySelector('input.dash-input-element') || node.querySelector('input');
  }

  function isLocationInput(node) {
    if (!node) {
      return false;
    }
    if (node.id === INPUT_ID) {
      return true;
    }
    return !!(node.closest && node.closest('#' + INPUT_ID));
  }

  function ensureHost(input) {
    var host = input.closest('.location-autocomplete-host');
    if (host) {
      return host;
    }
    var container = input.closest('.dash-input-container') || input.parentElement;
    if (!container) {
      return null;
    }
    container.classList.add('location-autocomplete-host');
    return container;
  }

  function renderDropdown(input, matches) {
    hideDropdown();
    if (!matches.length) {
      return;
    }

    var host = ensureHost(input);
    if (!host) {
      return;
    }

    dropdown = document.createElement('div');
    dropdown.className = 'location-autocomplete-dropdown';
    dropdown.setAttribute('role', 'listbox');

    matches.forEach(function (place, index) {
      var label = displayLabel(place);
      var item = document.createElement('button');
      item.type = 'button';
      item.className = 'location-autocomplete-item';
      item.textContent = label;
      item.setAttribute('role', 'option');
      item.addEventListener('mousedown', function (event) {
        event.preventDefault();
        selectSuggestion(input, label);
      });
      item.addEventListener('mouseenter', function () {
        setActiveIndex(index);
      });
      dropdown.appendChild(item);
    });

    host.appendChild(dropdown);
    setActiveIndex(-1);
  }

  function setActiveIndex(index) {
    activeIndex = index;
    if (!dropdown) {
      return;
    }
    var items = dropdown.querySelectorAll('.location-autocomplete-item');
    for (var i = 0; i < items.length; i++) {
      items[i].classList.toggle('active', i === activeIndex);
    }
  }

  function setInputValue(input, value) {
    if (!input) {
      return;
    }
    // Dash wraps dcc.Input in React; a plain `input.value = …` does not update
    // the component state, so Start/validation never see the selection.
    var descriptor = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      'value'
    );
    if (descriptor && descriptor.set) {
      descriptor.set.call(input, value);
    } else {
      input.value = value;
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function selectSuggestion(input, value) {
    if (!input) {
      return;
    }
    suppressNextInput = true;
    setInputValue(input, value);
    hideDropdown();
    input.focus();
  }

  function updateSuggestions(input) {
    if (!input) {
      hideDropdown();
      return;
    }
    if (suppressNextInput) {
      suppressNextInput = false;
      return;
    }

    var value = input.value || '';
    if (value.trim().length < MIN_CHARS) {
      hideDropdown();
      return;
    }

    loadPlaces().then(function () {
      var current = getInputElement();
      if (!current || document.activeElement !== current) {
        return;
      }
      renderDropdown(current, filterPlaces(current.value || ''));
    });
  }

  function scheduleUpdate(input) {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(function () {
      updateSuggestions(input || getInputElement());
    }, DEBOUNCE_MS);
  }

  function onInputKeyDown(event) {
    if (!isLocationInput(event.target) || !dropdown) {
      return;
    }
    var items = dropdown.querySelectorAll('.location-autocomplete-item');
    if (!items.length) {
      return;
    }

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActiveIndex(Math.min(activeIndex + 1, items.length - 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex(Math.max(activeIndex - 1, 0));
    } else if (event.key === 'Enter' && activeIndex >= 0) {
      event.preventDefault();
      selectSuggestion(getInputElement(), items[activeIndex].textContent);
    } else if (event.key === 'Escape') {
      hideDropdown();
    }
  }

  function attachAutocomplete(input) {
    if (!input || input.getAttribute('data-location-autocomplete') === '1') {
      return;
    }
    input.setAttribute('data-location-autocomplete', '1');
    input.setAttribute('autocomplete', 'off');
    ensureHost(input);
  }

  function scan() {
    var input = getInputElement();
    if (input) {
      attachAutocomplete(input);
    }
  }

  function refresh(pathname) {
    if (pathname && pathname !== '/startup') {
      hideDropdown();
      return;
    }
    scan();
    loadPlaces();
    var input = getInputElement();
    if (input && document.activeElement === input) {
      scheduleUpdate(input);
    }
  }

  document.addEventListener(
    'input',
    function (event) {
      if (!isLocationInput(event.target)) {
        return;
      }
      var input = getInputElement();
      attachAutocomplete(input);
      scheduleUpdate(input);
    },
    true
  );

  document.addEventListener(
    'focusin',
    function (event) {
      if (!isLocationInput(event.target)) {
        return;
      }
      var input = getInputElement();
      attachAutocomplete(input);
      scheduleUpdate(input);
    },
    true
  );

  document.addEventListener('keydown', onInputKeyDown, true);

  document.addEventListener(
    'focusout',
    function (event) {
      if (!isLocationInput(event.target)) {
        return;
      }
      setTimeout(hideDropdown, 150);
    },
    true
  );

  window.addEventListener('storage', function (event) {
    if (event.key === LOCALE_STORE_ID) {
      var input = getInputElement();
      if (input && document.activeElement === input) {
        scheduleUpdate(input);
      }
    }
  });

  window.sugarSugarLocationAutocomplete = {
    refresh: refresh,
    scan: scan,
  };

  document.addEventListener('DOMContentLoaded', function () {
    scan();
    loadPlaces();
  });

  var observer = new MutationObserver(function () {
    scan();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  scan();
  loadPlaces();
})();
