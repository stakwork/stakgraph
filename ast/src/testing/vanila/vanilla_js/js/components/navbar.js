export class Navbar {
  constructor() {
    this.container = document.getElementById("navbar-container");
  }

  render() {
    const currentPath =
      window.location.pathname.split("/").pop() || "index.html";

    this.container.innerHTML = `
            <div class="container">
                <nav>
                    <ul>
                        <li><a href="index.html" class="${currentPath === "index.html" ? "active" : ""}">Home</a></li>
                        <li><a href="about.html" class="${currentPath === "about.html" ? "active" : ""}">About</a></li>
                        <li><a href="contact.html" class="${currentPath === "contact.html" ? "active" : ""}">Contact</a></li>
                    </ul>
                </nav>
            </div>
        `;
  }
}
