export class Footer {
  constructor() {
    this.container = document.getElementById("footer-container");
  }

  render() {
    const year = new Date().getFullYear();
    this.container.innerHTML = `
            <div class="container" style="text-align: center;">
                <p>&copy; ${year} Vanilla JS App. All rights reserved.</p>
            </div>
        `;
  }
}
