{
  description = "connectome_utils - HiAER-Spike network model library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, devshell, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ devshell.overlays.default ]; };
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
      in {
        packages.default = p2n.mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python311;
          preferWheels = true;
        };

        devShells.default = pkgs.devshell.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python311;
            })
          ];
        };
      }
    );
}
