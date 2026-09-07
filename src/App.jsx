import AnalyzeAudio from "./pages/AnalyzeAudio";
import { useState } from "react";
import Home from "./pages/Home";
import Protection from "./pages/Protection";

function App() {
  const [currentPage, setCurrentPage] = useState("home");

  if (currentPage === "protection") {
    return (
      <Protection
        onEndProtection={() => setCurrentPage("home")}
      />
    );
  }

  if (currentPage === "analyze") {
  return (
    <AnalyzeAudio
      onBack={() => setCurrentPage("home")}
    />
  );

  }

  return (
    <Home
      onStartProtection={() =>
        setCurrentPage("protection")
      }
      onAnalyzeAudio={() =>
        setCurrentPage("analyze")
      }
    />
  );
}
export default App;