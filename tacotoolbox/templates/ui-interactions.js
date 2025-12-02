/**
 * TACO Documentation UI Interactions
 * 
 * Handles interactive features for the documentation page:
 * - Syntax highlighting initialization
 * - Code language tab switching (Python/R/Julia)
 * - Copy-to-clipboard buttons for code blocks
 */

document.addEventListener('DOMContentLoaded', function() {
    // Syntax highlighting
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
    
    // Code language tabs
    const codeTabs = document.querySelectorAll('.code-tab');
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            
            // Update active tab
            codeTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Show/hide corresponding code blocks
            codeBlocks.forEach(block => {
                block.style.display = block.getAttribute('data-lang') === lang ? 'block' : 'none';
            });
        });
    });
    
    // Copy buttons for code blocks
    document.querySelectorAll('pre:has(code)').forEach(pre => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = 
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
            '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
            '</svg><span>Copy</span>';
        
        button.addEventListener('click', () => {
            const code = pre.querySelector('code').textContent;
            
            navigator.clipboard.writeText(code).then(() => {
                // Show "Copied!" feedback
                button.classList.add('copied');
                button.innerHTML = 
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                    '<polyline points="20 6 9 17 4 12"></polyline>' +
                    '</svg><span>Copied!</span>';
                
                // Revert after 2 seconds
                setTimeout(() => {
                    button.classList.remove('copied');
                    button.innerHTML = 
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
                        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
                        '</svg><span>Copy</span>';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        });
        
        pre.appendChild(button);
    });
});