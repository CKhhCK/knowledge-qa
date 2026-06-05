/**
 * App — Root component with login gating.
 */
import React, { useState } from 'react';
import { useAuthStore } from './store/authStore';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ChatPage from './pages/ChatPage';

const App: React.FC = () => {
  const isLoggedIn = useAuthStore(s => s.isLoggedIn);
  const logout = useAuthStore(s => s.logout);
  const user = useAuthStore(s => s.user);
  const [showRegister, setShowRegister] = useState(false);

  if (!isLoggedIn) {
    if (showRegister) {
      return (
        <RegisterPage
          onRegister={() => setShowRegister(false)}
          onSwitchToLogin={() => setShowRegister(false)}
        />
      );
    }
    return (
      <LoginPage
        onLogin={() => {}}
        onSwitchToRegister={() => setShowRegister(true)}
      />
    );
  }

  return <ChatPage />;
};

export default App;
