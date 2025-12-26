INSERT INTO model (id, name) VALUES
  (1, 'laion/clap-htsat-unfused'),
  (2, 'laion/clap-htsat-fused'),
  (3, 'm-a-p/MERT-v1-95M'),
  (4, 'openl3-mel256-512')
ON CONFLICT(name) DO NOTHING;
