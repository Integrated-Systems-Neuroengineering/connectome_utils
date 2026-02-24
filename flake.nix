{
  description = "connectome_utils - HiAER-Spike network model library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
      in {
        packages.default = (p2n.mkPoetryPackages {
          projectDir = ./.;
          python = pkgs.python310;
        }).poetryPackage;

        devShells.default = pkgs.mkShell {
          packages = [
            (p2n.mkPoetryEnv {
              projectDir = ./.;
              python = pkgs.python310;
            })
          ];
        };
      }
    );
}
