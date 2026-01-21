{
  description = "uv + nix flake dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      py = pkgs.python312;

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

      mkEnv = ''
          export UV_PYTHON="${py}/bin/python"
          export UV_LINK_MODE=copy   # avoids symlink weirdness across stores/venvs

          # help compilation/linking discover nix libs
          export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPath "lib/pkgconfig" nativeLibs}"
      '';

      # cpu_shell = pkgs.mkShell {
      #   packages = buildInputs ++ nativeLibs;
      #   shellHook = mkEnv + ''export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}":$LD_LIBRARY_PATH'';
      # };

      # cuda_shell = pkgs.mkShell {
      #     packages = buildInputs ++ nativeLibs ++ cudaLibs;
      #     shellHook = mkEnv + ''export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}":"${pkgs.lib.makeLibraryPath cudaLibs}":$LD_LIBRARY_PATH'';
      # };

      makeShell = { cuda ? false }:
        let
          libs = nativeLibs ++ pkgs.lib.optionals cuda cudaLibs;
        in
          {
            inherit libs;
            packages = buildInputs ++ libs;
            shellHook = mkEnv + ''export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}":$LD_LIBRARY_PATH'';
          };
      
      mkShellFor = args: pkgs.mkShell (makeShell args);
      cpu_shell = mkShellFor { cuda = false; };
      cuda_shell = mkShellFor { cuda = true; };
          

      run-uvcellpose = pkgs.writeShellApplication {
        name = "run-cellpose";

        runtimeInputs = nativeLibs ++ cudaLibs;
        # Don't include the base LD_LIBRARY_PATH in the run version, it's not set and complains
        text = mkEnv + ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}":"${pkgs.lib.makeLibraryPath cudaLibs}"
          uv run cellpose
        '';
      };
    in
    {
      devShells.${system} = {
        gpu = cuda_shell;
        cpu = cpu_shell;
        default = cuda_shell;
      };

      apps.${system}.default = {
        type = "app";
        program = "${run-uvcellpose}/bin/run-cellpose";
      };
    };
}
