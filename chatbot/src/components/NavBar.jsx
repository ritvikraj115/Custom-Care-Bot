

import { Link } from "react-router-dom";
import logo from "../utils/logo.png";

export default function Navbar() {
  const isLoggedIn = !!localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/login";
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <img src={logo} alt="Logo" />
          <span>
            Custom <span className="navbar-logo-brand">Care</span>
          </span>
        </Link>
        <div className="navbar-links">
          <Link to="/dashboard" className="navbar-link">Dashboard</Link>
          <Link to="/create-bot" className="navbar-link">Create Bot</Link>
          {isLoggedIn ? (
            <Link className="navbar-link" onClick={handleLogout}>Logout</Link>
          ) : (
            <>
              <Link to="/login" className="navbar-link">Login</Link>
              <Link to="/register" className="navbar-link">Register</Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
