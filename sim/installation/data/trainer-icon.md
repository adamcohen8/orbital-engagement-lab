# RPO Trainer Desktop Icon

`trainer-icon.png` is the approved square adaptation of the public Trainer
start-screen artwork, `sim/game/assets/OEL_RPO_Trainer.png`. It retains the OEL
logo, satellite accents, and RPO TRAINER title, with the interactive prompts
and peripheral screen frame removed. The square adaptation was made with
imagegen and approved for OEL-wide use on 2026-09-28.

The PNG is the shared master and Linux launcher icon. `trainer-icon.ico`
contains 16, 24, 32, 48, 64, 128, and 256 pixel images for Windows;
`trainer-icon.icns` contains macOS icon representations through 1024 pixels.
These are packaged assets: installation needs no image conversion tools.

The macOS bundle copies the ICNS into `Contents/Resources`. Windows embeds
the ICO images in distlib's bare GUI launcher before its Python dispatch
payload is appended, and also assigns the ICO to the Start Menu shortcut.
Linux references a PNG copied into the stable managed launcher directory.
None of the desktop icon paths depend on a removable engine-version directory.
