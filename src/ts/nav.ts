

class TabNavigator_class {

    public switchToPage(page_id: string) {
        const pages = document.querySelectorAll('.page');
        for (const page of pages) {
            if (page.id === page_id)
                page.classList.remove('hidden');
            else
                page.classList.add('hidden');
        }
        const page_buttons = document.querySelectorAll('.page-switcher');
        for (const page_button of page_buttons)
            (<HTMLButtonElement>page_button).disabled = (page_button.getAttribute('page') === page_id);

    }

    constructor() {

        document.querySelectorAll('.page-switcher').forEach((button) => {
            (<HTMLButtonElement>button).addEventListener('click', () => {
                this.switchToPage(button.getAttribute('page') || '')
            })
        });
    }
}

export const TabNavigator = new TabNavigator_class();





