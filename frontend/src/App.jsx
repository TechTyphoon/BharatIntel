import { useState, useEffect } from "react";
import Header from "./components/Header";
import Executive from "./components/Executive";
import Headlines from "./components/Headlines";
import Section from "./components/Section";
import StatusBar from "./components/StatusBar";
import Loader from "./components/Loader";
import Empty from "./components/Empty";
import Settings from "./components/Settings";
import { getLatestBriefing, generateBrief, getStatus, getPdfUrl } from "./api";
import "./App.css";

const SECTION_ORDER = [
  "geopolitics",
  "technology",
  "indian_politics",
  "science",
  "civilisation",
  "editors_picks",
];

function sortSections(sections) {
  return [...sections].sort((a, b) => {
    const aKey = a.category.toLowerCase().replace(/\s+/g, "_");
    const bKey = b.category.toLowerCase().replace(/\s+/g, "_");
    const ai = SECTION_ORDER.indexOf(aKey);
    const bi = SECTION_ORDER.indexOf(bKey);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });
}

export default function App() {
  const [briefing, setBriefing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [statusMsg, setStatusMsg] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    fetchBriefing();
    checkRunningPipeline();
  }, []);

  async function checkRunningPipeline() {
    try {
      const st = await getStatus();
      if (st.pipeline_running) {
        setGenerating(true);
        setStatusMsg("Pipeline is already running… Waiting for it to finish.");
        pollUntilDone();
      }
    } catch {
      // ignore — fetchBriefing will handle errors
    }
  }

  async function pollUntilDone() {
    while (true) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const st = await getStatus();
        if (!st.pipeline_running) {
          if (st.last_error) {
            setError(st.last_error);
            setStatusMsg(null);
          } else {
            const result = st.last_result || {};
            if (result.status === "partial") {
              setStatusMsg(
                (result.message || "Briefing generated with partial content.") +
                  " ⚠️ Some sections may use fallback content — check Settings for API key issues."
              );
            } else {
              setStatusMsg(result.message || "Briefing generated successfully.");
            }
            await fetchBriefing();
          }
          setGenerating(false);
          return;
        }
      } catch {
        // keep polling
      }
    }
  }

  async function fetchBriefing() {
    setLoading(true);
    setError(null);
    try {
      const data = await getLatestBriefing();
      setBriefing(data);
    } catch (err) {
      if (err.message === "Failed to fetch") {
        setError(
          "Could not reach the server — it may be waking up (free tier cold start). Please try again in ~30 seconds. Click ⚙ Settings to configure your API keys."
        );
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    setStatusMsg("Running pipeline… This may take a few minutes.");
    try {
      const res = await generateBrief();
      // generateBrief() polls internally and returns when done
      if (res._partial) {
        setStatusMsg(
          (res.message || "Briefing generated with partial content.") +
            " ⚠️ Some sections may use fallback content — check Settings for API key issues."
        );
      } else {
        setStatusMsg(res.message || "Briefing generated successfully.");
      }
      await fetchBriefing();
    } catch (err) {
      if (err.message === "Pipeline is already running.") {
        setStatusMsg("Pipeline is already running… Waiting for it to finish.");
        pollUntilDone();
        return;
      }
      setError(err.message);
      setStatusMsg(null);
    } finally {
      setGenerating(false);
    }
  }

  function handleDownload() {
    window.open(getPdfUrl(), "_blank");
  }

  if (loading) return <Loader />;

  return (
    <div className="app">
      <Header
        briefingDate={briefing?.date}
        onGenerate={handleGenerate}
        onDownload={handleDownload}
        onSettings={() => setSettingsOpen(true)}
        loading={generating}
      />

      <Settings open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <StatusBar status={statusMsg} error={error} onOpenSettings={() => setSettingsOpen(true)} />

      <main className="main">
        {!briefing ? (
          <Empty />
        ) : (
          <>
            <Executive data={briefing.executive_summary} />
            <Headlines headlines={briefing.headlines} />
            <div className="sections">
              {sortSections(briefing.sections || []).map((section, i) => (
                <Section key={section.category || i} section={section} />
              ))}
            </div>
          </>
        )}
      </main>

      <footer className="footer">
        <p>BharatIntel — AI-Powered Intelligence Briefing System</p>
      </footer>
    </div>
  );
}
