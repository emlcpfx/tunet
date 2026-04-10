import { SignIn } from '@clerk/nextjs'

export default function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F9FAFB]">
      <div className="w-full max-w-md px-4">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2.5 mb-4">
            <svg viewBox="0 0 32 32" fill="none" className="w-10 h-10">
              <circle cx="16" cy="16" r="14" fill="#ae69f4"/>
              <path d="M12 10c0-1 1.5-2 4-2s4 1 4 2c0 2-4 3-4 5 0 1.5 0 2 0 2m0 3v1"
                stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span className="text-2xl font-bold text-[#111827]">TuNet Cloud</span>
          </div>
          <p className="text-sm text-[#6b7280]">Sign in to your account</p>
        </div>
        <SignIn
          appearance={{
            elements: {
              rootBox: 'w-full',
              card: 'shadow-none border border-[#e5e7eb] rounded-xl',
              headerTitle: 'hidden',
              headerSubtitle: 'hidden',
              socialButtonsBlockButton: 'border border-[#e5e7eb] rounded-lg',
              formButtonPrimary: 'bg-[#ae69f4] hover:bg-[#7E3AF2]',
              footerActionLink: 'text-[#ae69f4]',
            },
          }}
        />
      </div>
    </div>
  )
}
