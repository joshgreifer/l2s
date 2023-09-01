

class TabNavigator_class {

    public switchToPage(page_id: string) {
        const pages = document.querySelectorAll('.page');
        for (const page of pages) {
            if (page.id === page_id)
                page.classList.remove('hidden');
            else
                page.classList.add('hidden');
        }
        const page_buttons = document.querySelectorAll('.page-tab');
        for (const page_button of page_buttons)
            (<HTMLButtonElement>page_button).disabled = (page_button.getAttribute('page') === page_id);

    }

    constructor() {
        const tabContainer = document.querySelector('.page-tabs')
        if (tabContainer === null)
            window.alert("Can't construct tab navigator, couldn't find a tab container element in document (with the class'page-tabs'.")
        else {

            const pages = document.querySelectorAll('.page');
            for (const page of pages) {
                const tab_el = document.createElement('button');
                tab_el.className = 'page-tab';
                tab_el.setAttribute('page', page.id);
                tab_el.innerText = page.getAttribute('title') || page.id;
                tab_el.addEventListener('click', () => {
                    this.switchToPage(page.id) });
                tabContainer.appendChild(tab_el);
            }

        }

    }
}

export const TabNavigator = new TabNavigator_class();





