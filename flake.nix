{
  description = "Deterministic Nebulabrot renderer and its 4K PNG artifact";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/5e867b1db0f219e0e906574cca1175f421a87834";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python314;
      pythonPackages = python.pkgs;

      nebulabrot = pythonPackages.buildPythonApplication {
        pname = "nebulabrot";
        version = "0.1.0";
        format = "pyproject";

        src = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = pkgs.lib.fileset.unions [
            ./pyproject.toml
            ./README.md
            ./src
            ./tests
          ];
        };

        nativeBuildInputs = [
          pythonPackages.setuptools
          pkgs.ruff
          pkgs.pyrefly
        ];
        propagatedBuildInputs = [
          pythonPackages.numpy
          pythonPackages.pillow
        ];
        nativeCheckInputs = [ pythonPackages.pytestCheckHook ];
        doCheck = true;
        pythonImportsCheck = [ "nebulabrot" ];

        # Formatting, linting, type checking and tests are build requirements.
        preCheck = ''
          ruff format --check .
          ruff check .
          pyrefly check
        '';

        passthru.env = python.withPackages (
          ps: [
            ps.numpy
            ps.pillow
            ps.pytest
          ]
        );
      };

      # The artifact itself: `nix build .#image` yields the 4K PNG in the store.
      # `preferLocalBuild` keeps it out of the "just substitute it" path, since
      # the point of building it is that the bytes are the point.
      image = pkgs.runCommand "nebulabrot-3840x2160.png"
        {
          nativeBuildInputs = [ nebulabrot ];
          preferLocalBuild = true;
          allowSubstitutes = false;
        }
        ''
          nebulabrot --output $out --jobs "$NIX_BUILD_CORES"
        '';
    in
    {
      packages.${system} = {
        default = image;
        inherit nebulabrot image;
      };

      checks.${system}.default = nebulabrot;

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          nebulabrot
          nebulabrot.passthru.env
          pkgs.ruff
          pkgs.pyrefly
        ];
      };
    };
}
