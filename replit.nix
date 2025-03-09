{pkgs}: {
  deps = [
    pkgs.xsimd
    pkgs.libxcrypt
    pkgs.portaudio
    pkgs.libsndfile
    pkgs.glibcLocales
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.cairo
    pkgs.ffmpeg-full
  ];
}
