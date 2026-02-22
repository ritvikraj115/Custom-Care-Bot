import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import CreateBot from "./pages/CreateBot";
import BotDetail from "./pages/BotDetail";
import Navbar from "./components/NavBar";
import PublicChat from "./pages/PublicChat";

const isAuthenticated = () => {
  return !!localStorage.getItem("token");
};

const PrivateRoute = ({ children }) => {
  return isAuthenticated() ? children : <Navigate to="/login" />;
};

export default function App() {
  return (
    <Router>
      <Navbar />
      {isAuthenticated()}

      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />

        <Route
          path="/create-bot"
          element={
            <PrivateRoute>
              <CreateBot />
            </PrivateRoute>
          }
        />

        <Route
          path="/bots/:botId"
          element={
            <PrivateRoute>
              <BotDetail />
            </PrivateRoute>
          }
        />
        <Route
          path="/chat/:botId"
          element={
            <PrivateRoute>
              <PublicChat />
            </PrivateRoute>
          }
        />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
        <ToastContainer
    position="top-right"
    autoClose={4000}
    hideProgressBar={false}
    newestOnTop
    closeOnClick
    pauseOnHover
  />
    </Router>
  );
}
