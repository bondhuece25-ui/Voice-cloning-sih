import { useState } from "react";
import Home from "./pages/Home";
import Protection from "./pages/Protection";

function App() {
  const [currentPage, setCurrentPage] = useState("home");

  return currentPage === "home" ? (
    <Home onStartProtection={() => setCurrentPage("protection")} />
  ) : (
    <Protection onEndProtection={() => setCurrentPage("home")} />
  );
}

export default App;