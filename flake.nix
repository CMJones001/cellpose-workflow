{
  description = "A uv based flake to run cellpose with or without a GPU";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      py = pkgs.python310;

      buildInputs = [
        pkgs.uv
        py

        # common build tooling for sdists
        pkgs.pkg-config
        pkgs.gcc
        pkgs.gnumake
        pkgs.cmake
      ];

      nativeLibs = [
        pkgs.openssl
        pkgs.zlib
        pkgs.libffi
        pkgs.sqlite
        pkgs.stdenv.cc.cc.lib
        pkgs.zstd
        pkgs.glib
        pkgs.libGL

        # Qt text/font/render basics
        pkgs.fontconfig pkgs.freetype pkgs.harfbuzz pkgs.icu
        pkgs.zlib pkgs.libpng pkgs.libjpeg

        # common runtime plumbing
        pkgs.glib pkgs.dbus
        pkgs.openssl

        # X11 / xcb platform plugin
        pkgs.xorg.libX11
        pkgs.xorg.libxcb
        pkgs.xorg.xcbutil
        pkgs.xorg.xcbutilcursor
        pkgs.xorg.xcbutilimage
        pkgs.xorg.xcbutilkeysyms
        pkgs.xorg.xcbutilrenderutil
        pkgs.xorg.xcbutilwm
        pkgs.libxkbcommon

        pkgs.xorg.libXrender pkgs.xorg.libXi pkgs.xorg.libXext pkgs.xorg.libXfixes
        pkgs.xorg.libXrandr pkgs.xorg.libXcursor pkgs.xorg.libXinerama
        pkgs.xorg.libXdamage pkgs.xorg.libXcomposite
        pkgs.xorg.libSM pkgs.xorg.libICE
      ];

      cudaLibs = [
        pkgs.cudaPackages.cuda_cudart
        pkgs.cudaPackages.libcublas
        pkgs.cudaPackages.libcufft
        pkgs.cudaPackages.libcurand
        pkgs.cudaPackages.libcusolver
        pkgs.cudaPackages.libcusparse
        pkgs.cudaPackages.cudnn
        pkgs.cudaPackages.nccl
        # optional but often helpful:
        pkgs.cudaPackages.cuda_nvrtc
      ];

      extraPackages = [
        pkgs.fiji
      ];

      libaryShellHook = ''
          export UV_PYTHON="${py}/bin/python"
          export UV_LINK_MODE=copy   # avoids symlink weirdness across stores/venvs
      '';

      makeShell = { cuda ? false }:
        let
          libs = nativeLibs ++ pkgs.lib.optionals cuda cudaLibs;
        in
          pkgs.mkShell {
            inherit libs;
            packages = buildInputs ++ libs ++ extraPackages;
            shellHook = libaryShellHook + ''
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}":$LD_LIBRARY_PATH
                export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPath "lib/pkgconfig" libs}"
            '';
          };
      
      cpu_shell = makeShell { cuda = false; };
      gpu_shell = makeShell { cuda = true; };

      makeApp = { cuda ? false}:
        let
          libs = nativeLibs ++ pkgs.lib.optionals cuda cudaLibs;
          app = pkgs.writeShellApplication {
            name = "run-cellpose";
            runtimeInputs = nativeLibs ++ cudaLibs;

            # The LD_LIBRARY_PATH is slightly different in the nix run version,
            # as the old LD_LIBRARY_PATH is unset, so we don't include it
            text = libaryShellHook + ''
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}"
                export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPath "lib/pkgconfig" libs}"
                uv run cellpose
            '';
          };
        in
          { type = "app"; program = "${app}/bin/run-cellpose"; };

      run-cellpose-cpu = makeApp { cuda = false; };
      run-cellpose-gpu = makeApp { cuda = true; };

    in
    {
      devShells.${system} = {
        gpu = gpu_shell;
        cpu = cpu_shell;
        default = gpu_shell;
      };

      apps.${system} = {
        cpu = run-cellpose-cpu;
        gpu = run-cellpose-gpu;
        default = run-cellpose-gpu;
        fiji = { type = "app"; program = "${pkgs.fiji}/bin/fiji"; };
      };
    };
}
