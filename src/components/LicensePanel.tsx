import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Key, CheckCircle2, Crown, Star, Zap, Shield,
  ArrowRight, Copy, Check
} from 'lucide-react';
import type { LicenseInfo } from '../types';
import { cn } from '../utils/cn';

interface LicensePanelProps {
  license: LicenseInfo;
  onValidate: (key: string) => void;
}

const PLANS = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    color: 'border-border-dim bg-bg-secondary',
    icon: <Star className="h-5 w-5 text-text-muted" />,
    features: [
      'Basic video generation',
      'Up to 512px resolution',
      'Lite model only',
      'Watermark on output',
      'Community support',
    ],
    buttonLabel: 'Current Plan',
    buttonStyle: 'bg-bg-tertiary text-text-muted border border-border-dim cursor-default',
  },
  {
    name: 'Pro',
    price: '$29',
    period: 'one-time',
    color: 'border-accent-purple/30 bg-gradient-to-br from-accent-purple/5 to-accent-blue/5',
    icon: <Zap className="h-5 w-5 text-accent-purple" />,
    badge: 'Popular',
    features: [
      'All model tiers (Lite → Ultra)',
      'Up to 1280×720 resolution',
      'No watermark',
      'Priority support',
      'Lifetime updates',
      'Commercial use license',
    ],
    buttonLabel: 'Upgrade to Pro',
    buttonStyle: 'bg-gradient-to-r from-accent-purple to-accent-blue text-white shadow-lg shadow-accent-purple/25',
  },
  {
    name: 'Enterprise',
    price: '$99',
    period: 'one-time',
    color: 'border-accent-amber/30 bg-gradient-to-br from-accent-amber/5 to-orange-500/5',
    icon: <Crown className="h-5 w-5 text-accent-amber" />,
    features: [
      'Everything in Pro',
      'Multi-seat license (up to 5)',
      'Custom model fine-tuning',
      'API access',
      'Dedicated support',
      'White-label option',
    ],
    buttonLabel: 'Contact Sales',
    buttonStyle: 'bg-gradient-to-r from-accent-amber to-orange-500 text-white shadow-lg shadow-accent-amber/25',
  },
];

export default function LicensePanel({ license, onValidate }: LicensePanelProps) {
  const [licenseKey, setLicenseKey] = useState('');
  const [copied, setCopied] = useState(false);

  const handleActivate = () => {
    if (licenseKey.trim()) {
      onValidate(licenseKey.trim());
    }
  };

  const handleCopyMachineId = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto p-6 space-y-8">
        <div>
          <h2 className="text-xl font-semibold text-text-primary">License & Plans</h2>
          <p className="text-sm text-text-secondary mt-1">
            Manage your license key and explore available plans.
          </p>
        </div>

        {/* Current License Status */}
        <div className={cn(
          'rounded-xl border p-5',
          license.valid
            ? 'bg-accent-green/5 border-accent-green/20'
            : 'bg-bg-secondary border-border-dim'
        )}>
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <div className={cn(
                'flex h-10 w-10 items-center justify-center rounded-xl',
                license.valid ? 'bg-accent-green/20' : 'bg-bg-tertiary'
              )}>
                {license.valid ? (
                  <CheckCircle2 className="h-5 w-5 text-accent-green" />
                ) : (
                  <Key className="h-5 w-5 text-text-muted" />
                )}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-text-primary">
                  {license.valid ? `${license.tier.charAt(0).toUpperCase() + license.tier.slice(1)} License Active` : 'No License Key'}
                </h3>
                <p className="text-xs text-text-secondary mt-0.5">
                  {license.valid
                    ? `Expires: ${license.expires_at} · Key: ${license.key.slice(0, 8)}...`
                    : 'Enter a license key to unlock Pro features'}
                </p>
                {license.valid && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {license.features.map(f => (
                      <span key={f} className="rounded-full bg-accent-green/10 px-2 py-0.5 text-xs text-accent-green">
                        {f}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Activate License */}
        <div className="rounded-xl bg-bg-secondary border border-border-dim p-5 space-y-4">
          <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
            <Key className="h-4 w-4 text-text-muted" />
            Activate License Key
          </h3>
          <div className="flex gap-2">
            <input
              type="text"
              value={licenseKey}
              onChange={(e) => setLicenseKey(e.target.value)}
              placeholder="XXXXX-XXXXX-XXXXX-XXXXX"
              className="flex-1 rounded-lg bg-bg-tertiary border border-border-dim px-4 py-2.5 text-sm text-text-primary placeholder-text-muted font-mono tracking-wider focus:border-accent-purple focus:outline-none focus:ring-1 focus:ring-accent-purple/50"
            />
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleActivate}
              className="rounded-lg bg-gradient-to-r from-accent-purple to-accent-blue px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-accent-purple/25 hover:shadow-accent-purple/40 transition-shadow"
            >
              Activate
            </motion.button>
          </div>
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Shield className="h-3.5 w-3.5" />
            <span>Machine ID: a7f3e2d1-b8c4-4a5e-9f0d-1c2b3a4e5f6d</span>
            <button
              onClick={handleCopyMachineId}
              className="ml-1 text-accent-blue hover:underline flex items-center gap-1"
            >
              {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>

        {/* Pricing Plans */}
        <div>
          <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider mb-4">Available Plans</h3>
          <div className="grid grid-cols-3 gap-4">
            {PLANS.map((plan, i) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className={cn('relative rounded-xl border p-5', plan.color)}
              >
                {plan.badge && (
                  <span className="absolute -top-2.5 left-4 rounded-full bg-accent-purple px-2.5 py-0.5 text-xs font-semibold text-white">
                    {plan.badge}
                  </span>
                )}
                <div className="flex items-center gap-2 mb-3">
                  {plan.icon}
                  <h4 className="text-sm font-semibold text-text-primary">{plan.name}</h4>
                </div>
                <div className="mb-4">
                  <span className="text-2xl font-bold text-text-primary">{plan.price}</span>
                  <span className="text-xs text-text-muted ml-1">/{plan.period}</span>
                </div>
                <ul className="space-y-2 mb-5">
                  {plan.features.map(f => (
                    <li key={f} className="flex items-start gap-2 text-xs text-text-secondary">
                      <CheckCircle2 className="h-3.5 w-3.5 text-accent-green flex-shrink-0 mt-0.5" />
                      {f}
                    </li>
                  ))}
                </ul>
                <button className={cn('w-full rounded-lg py-2 text-xs font-medium transition-all flex items-center justify-center gap-1', plan.buttonStyle)}>
                  {plan.buttonLabel}
                  {plan.name !== 'Free' && <ArrowRight className="h-3 w-3" />}
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
