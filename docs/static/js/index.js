(() => {
  'use strict';

  function initialize() {
    initializeCitationCopy();
    initializeFigureDialog();
    initializeDomainQuiz();
    initializeBackToTop();
  }

  function initializeCitationCopy() {
    const buttons = document.querySelectorAll('[data-copy-bibtex]');
    const citation = document.getElementById('bibtex-content');
    const status = document.getElementById('copy-status');
    if (!citation || !buttons.length) return;

    if (status) {
      status.setAttribute('role', 'status');
      status.setAttribute('aria-live', 'polite');
      status.setAttribute('aria-atomic', 'true');
    }

    buttons.forEach((button) => {
      const originalLabel = button.textContent;
      let restoreTimer;
      let copying = false;

      button.addEventListener('click', async (event) => {
        event.preventDefault();
        if (copying) return;
        copying = true;
        window.clearTimeout(restoreTimer);
        if (status) status.textContent = '';

        try {
          const text = citation.textContent.trim();
          if (!text) throw new Error('The citation is empty.');
          let copied = false;

          if (window.isSecureContext && navigator.clipboard?.writeText) {
            try {
              await navigator.clipboard.writeText(text);
              copied = true;
            } catch {
              // Local previews and browser permissions may require the fallback.
            }
          }

          if (!copied) copied = copyWithSelection(text);
          if (!copied) throw new Error('Clipboard access is unavailable.');

          button.textContent = 'Copied!';
          if (status) status.textContent = 'BibTeX copied to clipboard.';
        } catch {
          button.textContent = 'Copy unavailable';
          if (status) {
            status.textContent = 'Could not copy automatically. Select the BibTeX below and copy it manually.';
          }
        } finally {
          copying = false;
          restoreTimer = window.setTimeout(() => {
            button.textContent = originalLabel;
          }, 2400);
        }
      });
    });
  }

  function copyWithSelection(text) {
    const activeElement = document.activeElement;
    const selection = window.getSelection();
    const savedRanges = [];
    if (selection) {
      for (let index = 0; index < selection.rangeCount; index += 1) {
        savedRanges.push(selection.getRangeAt(index).cloneRange());
      }
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.readOnly = true;
    textarea.tabIndex = -1;
    textarea.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;padding:0;border:0;opacity:0;font-size:16px;';
    document.body.append(textarea);

    try {
      textarea.focus({ preventScroll: true });
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);
      return Boolean(document.execCommand?.('copy'));
    } finally {
      textarea.remove();
      if (activeElement instanceof HTMLElement && activeElement.isConnected) {
        activeElement.focus({ preventScroll: true });
      }
      if (selection) {
        selection.removeAllRanges();
        savedRanges.forEach((range) => selection.addRange(range));
      }
    }
  }

  function initializeFigureDialog() {
    const dialog = document.getElementById('figure-dialog');
    const image = document.getElementById('figure-dialog-image');
    const caption = document.getElementById('figure-dialog-caption');
    if (!dialog || !image || !caption || typeof dialog.showModal !== 'function') {
      // The figure links remain ordinary links when native dialogs are unavailable.
      return;
    }

    let opener = null;
    let previousOverflow = '';
    let previousOverflowPriority = '';
    let scrollLocked = false;

    document.querySelectorAll('a[data-zoom]').forEach((link) => {
      link.addEventListener('click', (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        if (dialog.open) return;

        image.src = link.href;
        caption.textContent = link.dataset.caption || link.querySelector('img')?.alt || 'Figure';
        image.alt = link.querySelector('img')?.alt || caption.textContent;

        try {
          dialog.showModal();
        } catch {
          // Let the browser follow the original image link if opening fails.
          return;
        }

        event.preventDefault();
        opener = link;
        previousOverflow = document.body.style.getPropertyValue('overflow');
        previousOverflowPriority = document.body.style.getPropertyPriority('overflow');
        document.body.style.setProperty('overflow', 'hidden');
        scrollLocked = true;
      });
    });

    dialog.querySelectorAll('[data-close-dialog]').forEach((button) => {
      button.addEventListener('click', (event) => {
        event.preventDefault();
        dialog.close();
      });
    });

    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      const outside = event.clientX < bounds.left || event.clientX > bounds.right
        || event.clientY < bounds.top || event.clientY > bounds.bottom;
      if (outside) dialog.close();
    });

    // Native Escape handling closes the dialog and follows the same cleanup path.
    dialog.addEventListener('close', () => {
      if (scrollLocked) {
        if (previousOverflow) {
          document.body.style.setProperty('overflow', previousOverflow, previousOverflowPriority);
        } else {
          document.body.style.removeProperty('overflow');
        }
        scrollLocked = false;
      }
      if (opener?.isConnected) opener.focus({ preventScroll: true });
      opener = null;
    });
  }

  function initializeDomainQuiz() {
    const form = document.getElementById('domain-quiz');
    const feedback = document.getElementById('domain-feedback');
    if (!form || !feedback) return;

    feedback.setAttribute('role', 'status');
    feedback.setAttribute('aria-live', 'polite');
    feedback.setAttribute('aria-atomic', 'true');

    function clearFeedback() {
      feedback.textContent = '';
      delete feedback.dataset.result;
      delete form.dataset.result;
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const selected = form.querySelector('input[name="domain"]:checked');
      if (!selected) {
        clearFeedback();
        feedback.textContent = 'Choose one option, then check your answer.';
        form.querySelector('input[name="domain"]')?.focus();
        return;
      }

      const correct = selected.value === 'C';
      form.dataset.result = String(correct);
      feedback.dataset.result = String(correct);
      feedback.textContent = correct
        ? 'Correct — C (전관용역) belongs to General society. The other four terms belong to Information and communication.'
        : 'Not quite. The answer is C (전관용역), which belongs to General society. The other four terms belong to Information and communication. You can choose again and retry.';
    });

    form.addEventListener('change', (event) => {
      if (event.target.matches('input[name="domain"]')) clearFeedback();
    });

    form.addEventListener('reset', () => {
      // Clear after the browser restores controls, including an <output> element.
      queueMicrotask(clearFeedback);
    });
  }

  function initializeBackToTop() {
    const buttons = document.querySelectorAll('[data-back-top]');
    if (!buttons.length) return;
    const hero = document.querySelector('#hero, [data-hero], .hero');
    let updatePending = false;

    function updateVisibility() {
      const visible = hero ? hero.getBoundingClientRect().bottom < 0 : window.scrollY > 400;
      buttons.forEach((button) => {
        button.hidden = !visible;
      });
      updatePending = false;
    }

    function requestVisibilityUpdate() {
      if (updatePending) return;
      updatePending = true;
      window.requestAnimationFrame(updateVisibility);
    }

    buttons.forEach((button) => {
      button.addEventListener('click', (event) => {
        event.preventDefault();
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        window.scrollTo({ top: 0, behavior: reducedMotion ? 'instant' : 'smooth' });
      });
    });

    window.addEventListener('scroll', requestVisibilityUpdate, { passive: true });
    window.addEventListener('resize', requestVisibilityUpdate, { passive: true });
    updateVisibility();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
