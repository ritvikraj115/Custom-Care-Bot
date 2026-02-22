import axios from "../api/axios";
import { toast } from "react-toastify";
import { getErrorMessage } from "../utils/getErrorMessage";

export default function Login() {
  const submit = async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));

    try {
      const res = await axios.post("/auth/login", data);
      localStorage.setItem("token", res.data.token);
      toast.success("Signed in successfully");
      window.location.href = "/";
    } catch (error) {
      const status = error.response?.status;

      if (status === 400) {
        toast.error("Invalid email or password");
      } else if (status === 401) {
        toast.error("You are not authorized to sign in");
      } else {
        toast.error(getErrorMessage(error, "Unable to sign in"));
      }
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-layout">
        <div className="page-header">
          <h2 className="page-title">Sign in to your account</h2>
          <p className="page-subtitle">
            Access your organization's AI assistants and knowledge base.
          </p>
        </div>

        <div className="card auth-card">
          <form onSubmit={submit}>
            <label className="form-label">Email address</label>
            <input
              name="email"
              placeholder="you@company.com"
              required
            />

            <label className="form-label">Password</label>
            <input
              name="password"
              type="password"
              placeholder="Enter your password"
              required
            />

            <button className="btn-block">Sign in</button>
          </form>
        </div>

        <p className="auth-footer muted">
          Don't have an account? <a href="/register">Create one</a>
        </p>
      </div>
    </div>
  );
}
