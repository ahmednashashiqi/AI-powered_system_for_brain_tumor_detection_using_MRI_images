// static/js/auth.js
import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  signInWithRedirect,
  getRedirectResult
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";

// Use your existing Firebase settings
const firebaseConfig = {
  apiKey: "AIzaSyCso1ATC1zQRT7tPKSC6ezLeI5jhQlKrXU",
  authDomain: "mri-project-e4465.firebaseapp.com",
  databaseURL: "https://mri-project-e4465-default-rtdb.firebaseio.com",
  projectId: "mri-project-e4465",
  storageBucket: "mri-project-e4465.firebasestorage.app",
  messagingSenderId: "171030013607",
  appId: "1:171030013607:web:1de6059c7dfc4bbf109d82",
  measurementId: "G-LXMMWWPECD"
};

// Prevent re-initialization
const app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
const auth = getAuth(app);

const $ = (s) => document.querySelector(s);
const emailEl   = $("#email");
const passEl    = $("#password");
const loginBtn  = $("#login");
const signupBtn = $("#signup");
const googleBtn = $("#google");
const msgEl     = $("#msg");

function msg(t, err=false){
  if (!msgEl) return;
  msgEl.textContent = t || "";
  msgEl.style.color = err ? "#f87171" : "#22d3ee";
}
function busy(v){
  [loginBtn, signupBtn, googleBtn].forEach(b => b && (b.disabled = v));
}

// Handle redirect result (in case popup was blocked)
(async ()=>{
  try{
    const rr = await getRedirectResult(auth);
    if (rr?.user) await sendToken(rr.user);
  }catch(e){ /* ignore */ }
})();

// --- Helper: retry once if token is "used too early" ---
async function postVerify(idToken, attempt=1){
  const res = await fetch("/auth/verify", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ idToken }),
    credentials: "include" // مهم لضمان تخزين الكوكي
  });

  let data = {};
  try { data = await res.json(); } catch(_) { data = { ok:false, error:`HTTP ${res.status}` }; }

  const err = (data?.error || "").toLowerCase();
  if (!res.ok && attempt === 1 && (err.includes("token used too early") || err.includes("clock"))) {
    await new Promise(r => setTimeout(r, 1500));  // انتظر 1.5 ثانية
    return postVerify(idToken, 2);
  }
  return { res, data };
}

// Send ID token to backend to create a session cookie
async function sendToken(user){
  const idToken = await user.getIdToken(/* forceRefresh = */ true);
  const { res, data } = await postVerify(idToken, 1);

  if (res.ok && data.ok){
    msg("Signed in successfully ✅");
    await new Promise(r => setTimeout(r, 50)); // فسحة لتثبيت الكوكي
    window.location.replace("/analyze");
  } else {
    msg(data.error || `Verification failed (HTTP ${res.status})`, true);
  }
}

// Wire up buttons (only if elements exist)
if (loginBtn && signupBtn && googleBtn && emailEl && passEl) {
  loginBtn.addEventListener("click", async ()=>{
    const email = emailEl.value.trim();
    const pass  = passEl.value;
    if (!email || !pass) return msg("Please enter email and password", true);
    try{
      busy(true);
      const cred = await signInWithEmailAndPassword(auth, email, pass);
      await sendToken(cred.user);
    } catch(e){
      msg(e.message || "Login failed", true);
    } finally { busy(false); }
  });

  signupBtn.addEventListener("click", async ()=>{
    const email = emailEl.value.trim();
    const pass  = passEl.value;
    if (!email || !pass) return msg("Please enter email and password", true);
    try{
      busy(true);
      const cred = await createUserWithEmailAndPassword(auth, email, pass);
      await sendToken(cred.user);
    } catch(e){
      msg(e.message || "Account creation failed", true);
    } finally { busy(false); }
  });

  googleBtn.addEventListener("click", async ()=>{
    try{
      busy(true);
      const provider = new GoogleAuthProvider();
      try {
        const cred = await signInWithPopup(auth, provider);
        await sendToken(cred.user);
      } catch(e){
        if (e?.code === "auth/popup-blocked" || e?.code === "auth/popup-closed-by-user") {
          await signInWithRedirect(auth, provider);
        } else {
          throw e;
        }
      }
    } catch(e){
      msg(e.message || "Google sign-in failed", true);
    } finally { busy(false); }
  });
}
