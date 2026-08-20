// Location autocomplete for #location-input on /startup.
//
// Loading policy: the suggestion corpus is fetched ONLY once the user has typed
// into the location field. It used to be a single 824 KB all-locales file
// fetched and parsed eagerly on every page of the app -- before consent, on the
// chart, on /faq -- which cost real seconds of main thread on low-spec phones.
// It is now one compact per-locale file (~50-80 KB, ~20 KB gzipped) built by
// `uv run build-locations`, requested on first keystroke in the field and then
// cached in memory for the page's lifetime.

(function () {
  'use strict';

  var INPUT_ID = 'location-input';
  var LOCALE_STORE_ID = 'interface-language';
  var DATA_URL_PREFIX = '/assets/location-suggestions.';
  var MIN_CHARS = 2;
  var MAX_RESULTS = 8;
  var DEBOUNCE_MS = 120;
  var SUPPORTED_LOCALES = ['de', 'en', 'es', 'fr', 'ro', 'ru', 'uk', 'zh'];

  // Keyed by locale: switching the UI language loads that language's file and
  // keeps the previous one, so switching back is free.
  var placesByLocale = {};
  var loadPromises = {};
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

  // Row shape from build_locations.py: [label, rank] or [label, rank, extras].
  // The lowercase and folded forms of the label are derived here rather than
  // stored, which is most of why the per-locale files are small.
  function expandRow(row) {
    if (!row || !row.length) {
      return null;
    }
    var label = String(row[0]);
    var lower = label.toLowerCase();
    var folded = asciiFold(label);
    var tokens = folded === lower ? [lower] : [lower, folded];
    var extras = row[2];
    if (extras && extras.length) {
      for (var i = 0; i < extras.length; i++) {
        var extra = String(extras[i]).toLowerCase();
        if (tokens.indexOf(extra) === -1) {
          tokens.push(extra);
        }
      }
    }
    return {
      label: label,
      rank: typeof row[1] === 'number' ? row[1] : 1000,
      tokens: tokens,
    };
  }

  function loadPlaces(locale) {
    if (placesByLocale[locale]) {
      return Promise.resolve(placesByLocale[locale]);
    }
    if (loadPromises[locale]) {
      return loadPromises[locale];
    }
    loadPromises[locale] = fetch(DATA_URL_PREFIX + locale + '.json')
      .then(function (response) {
        if (!response.ok) {
          throw new Error('Failed to load location suggestions');
        }
        return response.json();
      })
      .then(function (data) {
        var rows = [];
        if (Array.isArray(data)) {
          for (var i = 0; i < data.length; i++) {
            var place = expandRow(data[i]);
            if (place) {
              rows.push(place);
            }
          }
        }
        placesByLocale[locale] = rows;
        return rows;
      })
      .catch(function () {
        placesByLocale[locale] = [];
        return placesByLocale[locale];
      });
    return loadPromises[locale];
  }

  function placeMatches(place, q, qFold) {
    var tokens = place.tokens;
    for (var i = 0; i < tokens.length; i++) {
      if (tokens[i].indexOf(q) === 0 || (qFold && tokens[i].indexOf(qFold) === 0)) {
        return 2;
      }
    }
    for (var j = 0; j < tokens.length; j++) {
      if (tokens[j].indexOf(q) !== -1 || (qFold && tokens[j].indexOf(qFold) !== -1)) {
        return 1;
      }
    }
    return 0;
  }

  function placeSortKey(a, b) {
    if (a.rank !== b.rank) {
      return a.rank - b.rank;
    }
    return a.label.localeCompare(b.label);
  }

  function filterPlaces(places, query) {
    if (!places || !places.length || !query) {
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
      if (seen[place.label]) {
        continue;
      }
      seen[place.label] = true;
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
      var item = document.createElement('button');
      item.type = 'button';
      item.className = 'location-autocomplete-item';
      item.textContent = place.label;
      item.setAttribute('role', 'option');
      item.addEventListener('mousedown', function (event) {
        event.preventDefault();
        selectSuggestion(input, place.label);
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

    // First fetch of the corpus happens HERE -- the user is typing a location,
    // which is the only moment the data is worth its bytes.
    var locale = getLocale();
    loadPlaces(locale).then(function (places) {
      var current = getInputElement();
      if (!current || document.activeElement !== current) {
        return;
      }
      var query = current.value || '';
      var matches = filterPlaces(places, query);
      var trimmed = query.trim();
      // Exact match (e.g. "Erdenet, Mongolia"): close the list. On Android
      // Chrome the open panel plus min-height:100vh read as a blank white
      // slab between the field and the keyboard.
      if (matches.length === 1 && matches[0].label.toLowerCase() === trimmed.toLowerCase()) {
        hideDropdown();
        return;
      }
      renderDropdown(current, matches);
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

  // Called by the clientside callback on navigation. Deliberately does NOT
  // preload the corpus: arriving on /startup is not typing a location. There is
  // no MutationObserver either -- the delegated focus/input listeners below
  // attach on demand, so nothing runs while the user is elsewhere in the app.
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
})();
