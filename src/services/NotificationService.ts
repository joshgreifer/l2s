import { ui } from "../UI";

export class NotificationService {
  private notificationDiv: HTMLDivElement;

  constructor(div: HTMLDivElement = ui.notificationDiv) {
    this.notificationDiv = div;
  }

  notify(message: string): void {
    this.notificationDiv.innerText = message;
    this.notificationDiv.style.opacity = "1";

    window.setTimeout(() => {
      this.notificationDiv.style.opacity = "0";
    }, 5000);
  }
}

export const notifications = new NotificationService();
