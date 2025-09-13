/**
 * @file nav.ts
 * @author Josh Greifer <joshgreifer@gmail.com>
 * @copyright © 2020 Josh Greifer
 * @licence
 * Copyright © 2020 Josh Greifer
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:

 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * @summary Tabbed page.
 * This is part of my Typescript starter application, which provides a
 * boilerplate and utility code for custom HTML tags, simple paging, and API access.
 *
 * The starter application was designed to be as lean as possible - its only dependency
 * is eventemitter3.
 *
 * @example
 * Create an empty div with 'page-tabs' class:
 * <div class="page-tabs"></div>
 * <div class=
 *
 *
 */

import { ui } from "../UI";

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
        const tabContainer = ui.pageTabs;
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





